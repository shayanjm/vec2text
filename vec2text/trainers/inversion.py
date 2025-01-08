import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from vec2text.trainers.base import BaseTrainer

class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        # Learnable log-variance parameters for each loss term
        self.log_sigma_ce = nn.Parameter(torch.tensor(0.0))         # For cross-entropy loss
        self.log_sigma_cosine = nn.Parameter(torch.tensor(0.0))     # For cosine embedding loss
        self.log_sigma_contrastive = nn.Parameter(torch.tensor(0.0))# For contrastive loss
        self.log_sigma_cycle = nn.Parameter(torch.tensor(0.0))      # For cycle-consistency loss

    def forward(
        self,
        ce_loss: torch.Tensor,
        cosine_loss: torch.Tensor,
        contrastive_loss: torch.Tensor,
        cycle_loss: torch.Tensor
    ):
        """
        Each input is a scalar tensor representing the loss for a given objective.
        We apply uncertainty weighting to all four losses and sum them.
        """
        # Convert log-sigmas into sigmas
        sigma_ce = torch.exp(self.log_sigma_ce)
        sigma_cosine = torch.exp(self.log_sigma_cosine)
        sigma_contrastive = torch.exp(self.log_sigma_contrastive)
        sigma_cycle = torch.exp(self.log_sigma_cycle)

        # Weighted + log(sigma) penalty
        weighted_ce          = ce_loss          / (2 * sigma_ce**2)          + self.log_sigma_ce
        weighted_cosine      = cosine_loss      / (2 * sigma_cosine**2)      + self.log_sigma_cosine
        weighted_contrastive = contrastive_loss / (2 * sigma_contrastive**2) + self.log_sigma_contrastive
        weighted_cycle       = cycle_loss       / (2 * sigma_cycle**2)       + self.log_sigma_cycle

        # Total
        total_loss = weighted_ce + weighted_cosine + weighted_contrastive + weighted_cycle

        loss_info = {
            'weighted_ce_loss': weighted_ce.detach().item(),
            'weighted_cosine_loss': weighted_cosine.detach().item(),
            'weighted_contrastive_loss': weighted_contrastive.detach().item(),
            'weighted_cycle_loss': weighted_cycle.detach().item(),

            'log_sigma_ce': self.log_sigma_ce.detach().item(),
            'log_sigma_cosine': self.log_sigma_cosine.detach().item(),
            'log_sigma_contrastive': self.log_sigma_contrastive.detach().item(),
            'log_sigma_cycle': self.log_sigma_cycle.detach().item(),
        }
        return total_loss, loss_info


# ------------------------------------------------------------------------------
# Helpers for contrastive loss & cycle consistency
# ------------------------------------------------------------------------------
def compute_inbatch_contrastive_loss(
    pred_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standard in-batch contrastive (InfoNCE):
     - We want pred[i] to align with target[i] more than target[j] for j != i.
    """
    # Normalize for stable cosine similarities
    pred_norm = F.normalize(pred_embeddings, p=2, dim=-1)
    targ_norm = F.normalize(target_embeddings, p=2, dim=-1)

    # Similarity matrix => (batch_size, batch_size)
    sim_matrix = torch.matmul(pred_norm, targ_norm.t()) 
    sim_matrix = sim_matrix / temperature  # scale by temperature

    batch_size = sim_matrix.size(0)
    labels = torch.arange(batch_size, device=pred_embeddings.device)
    # CrossEntropy where row i's "correct" label is i
    contrastive_loss = nn.CrossEntropyLoss()(sim_matrix, labels)
    return contrastive_loss


def compute_cycle_consistency_loss(
    model: nn.Module,
    pred_embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    2nd forward pass:
      1) Use the predicted embeddings as `frozen_embeddings`.
      2) Re-decode and compare to the same ground-truth `labels`.
    Returns a CE loss for that second pass.
    """
    with torch.enable_grad():
        cycle_outputs = model(
            embedder_input_ids=None,
            embedder_attention_mask=None,
            labels=labels,
            frozen_embeddings=pred_embeddings.detach(),
        )
    cycle_ce_loss = cycle_outputs.loss
    return cycle_ce_loss


# ------------------------------------------------------------------------------
# Trainer Implementation
# ------------------------------------------------------------------------------
class InversionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        # Extended UncertaintyLoss with 4 log_sigma params
        self.uncertainty_loss = UncertaintyLoss()

        # Per-loss running means
        self.ce_running_mean = 1.0
        self.cosine_running_mean = 1.0
        self.contrastive_running_mean = 1.0
        self.cycle_running_mean = 1.0

        self.ce_batch_count = 0
        self.cosine_batch_count = 0
        self.contrastive_batch_count = 0
        self.cycle_batch_count = 0

        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model
        self.embedder = self.model.embedder

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        ce_loss = outputs.loss  # Cross-entropy

        params = list(self.model.parameters()) + list(self.uncertainty_loss.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=1e-3)

        # ~~~~~ Re-embed predicted text to get pred_embeddings ~~~~~
        logits = outputs.get("logits")
        pred_ids = torch.argmax(logits, dim=-1).detach().cpu()
        pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        pred_embeddings = []
        for text in pred_texts:
            pred_inputs = self.embedder_tokenizer(
                [text], return_tensors="pt", padding=True, truncation=True
            ).to(ce_loss.device)
            embedding = self.call_embedding_model(
                input_ids=pred_inputs["input_ids"],
                attention_mask=pred_inputs["attention_mask"],
            )
            pred_embeddings.append(embedding)
        pred_embeddings = torch.stack(pred_embeddings)  # (B, embed_dim)

        target_embeddings = inputs["frozen_embeddings"]
        if pred_embeddings.shape != target_embeddings.shape:
            pred_embeddings = pred_embeddings.view_as(target_embeddings)

        # ~~~~~ Cosine embedding loss ~~~~~
        cosine_loss = 1.0 - nn.functional.cosine_similarity(
            pred_embeddings, target_embeddings, dim=-1
        ).mean()

        # ~~~~~ Contrastive loss (in-batch) ~~~~~
        contrastive_loss = compute_inbatch_contrastive_loss(pred_embeddings, target_embeddings)

        # ~~~~~ Cycle consistency loss ~~~~~
        cycle_loss = compute_cycle_consistency_loss(
            model=model,
            pred_embeddings=pred_embeddings,
            labels=inputs["labels"],
        )

        # ~~~~~ Update running means ~~~~~
        self.ce_batch_count += 1
        self.ce_running_mean = (
            (self.ce_running_mean * (self.ce_batch_count - 1) + ce_loss.item())
            / self.ce_batch_count
        )

        self.cosine_batch_count += 1
        self.cosine_running_mean = (
            (self.cosine_running_mean * (self.cosine_batch_count - 1) + cosine_loss.item())
            / self.cosine_batch_count
        )

        self.contrastive_batch_count += 1
        self.contrastive_running_mean = (
            (self.contrastive_running_mean * (self.contrastive_batch_count - 1) + contrastive_loss.item())
            / self.contrastive_batch_count
        )

        self.cycle_batch_count += 1
        self.cycle_running_mean = (
            (self.cycle_running_mean * (self.cycle_batch_count - 1) + cycle_loss.item())
            / self.cycle_batch_count
        )

        # ~~~~~ Normalize each individually ~~~~~
        ce_norm          = ce_loss          / self.ce_running_mean
        cosine_norm      = cosine_loss      / self.cosine_running_mean
        contrastive_norm = contrastive_loss / self.contrastive_running_mean
        cycle_norm       = cycle_loss       / self.cycle_running_mean

        # ~~~~~ Uncertainty weighting across 4 terms ~~~~~
        total_loss, loss_info = self.uncertainty_loss(
            ce_norm,
            cosine_norm,
            contrastive_norm,
            cycle_norm
        )

        # ~~~~~ Log ~~~~~
        self.log({
            "ce_loss": ce_loss.detach().item(),
            "cosine_loss": cosine_loss.detach().item(),
            "contrastive_loss": contrastive_loss.detach().item(),
            "cycle_loss": cycle_loss.detach().item(),

            "ce_running_mean": self.ce_running_mean,
            "cosine_running_mean": self.cosine_running_mean,
            "contrastive_running_mean": self.contrastive_running_mean,
            "cycle_running_mean": self.cycle_running_mean,

            "normalized_ce_loss": ce_norm.detach().item(),
            "normalized_cosine_loss": cosine_norm.detach().item(),
            "normalized_contrastive_loss": contrastive_norm.detach().item(),
            "normalized_cycle_loss": cycle_norm.detach().item(),
            "total_loss": total_loss.detach().item(),
            **loss_info,
        })

        return (total_loss, outputs) if return_outputs else total_loss

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Optionally override to log data-specific metrics. 
        """
        self._compute_data_metrics(inputs=inputs)  # <-- if you have a custom data metric
        return super().training_step(model, inputs)

    def evaluation_loop(self, *args, **kwargs) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Override to compute ppl from eval loss.
        """
        output = super().evaluation_loop(*args, **kwargs)
        metric_key_prefix = kwargs["metric_key_prefix"]
        try:
            perplexity = math.exp(output.metrics[f"{metric_key_prefix}_loss"])
        except KeyError:
            perplexity = -1
        except OverflowError:
            perplexity = float("inf")
        output.metrics[f"{metric_key_prefix}_perplexity"] = perplexity
        return output

    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """
        Rename old keys for backward compatibility if needed.
        """
        if {
            "embedding_transform.2.weight",
            "embedding_transform.2.bias",
        } <= state_dict.keys():
            print(
                "Renaming keys",
                {"embedding_transform.2.weight", "embedding_transform.2.bias"},
                "for backward compatibility.",
            )
            state_dict["embedding_transform.3.weight"] = state_dict.pop(
                "embedding_transform.2.weight"
            )
            state_dict["embedding_transform.3.bias"] = state_dict.pop(
                "embedding_transform.2.bias"
            )
        return state_dict
