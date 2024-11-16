import math
from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import transformers

from vec2text.trainers.base import BaseTrainer

class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        # Learnable log-variance parameters for uncertainty weighting
        self.log_sigma_ce = nn.Parameter(torch.tensor(0.0))  # For cross-entropy loss
        self.log_sigma_embedding = nn.Parameter(torch.tensor(0.0))  # For embedding loss

    def forward(self, ce_loss, embedding_loss):
        # Compute uncertainties
        sigma_ce = torch.exp(self.log_sigma_ce)
        sigma_embedding = torch.exp(self.log_sigma_embedding)

        # Weighted total loss
        weighted_ce_loss = ce_loss / (2 * sigma_ce**2) + self.log_sigma_ce
        weighted_embedding_loss = embedding_loss / (2 * sigma_embedding**2) + self.log_sigma_embedding

        # Combine losses
        total_loss = weighted_ce_loss + weighted_embedding_loss

        return total_loss, {
            'weighted_ce_loss': weighted_ce_loss.detach().item(),
            'weighted_embedding_loss': weighted_embedding_loss.detach().item(),
            'log_sigma_ce': self.log_sigma_ce.detach().item(),
            'log_sigma_embedding': self.log_sigma_embedding.detach().item(),
        }

class InversionTrainer(BaseTrainer):
    def __init__(
        self, *args, max_cache_size=10000, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.uncertainty_loss = UncertaintyLoss()

        # New parameters: maximum cache size and embedding loss interval
        self.max_cache_size = max_cache_size
        self.embedding_loss_count = 0  # Track number of accumulated steps
        self.ce_running_mean = 1.0  # Initialize running mean for ce_loss
        self.ce_batch_count = 0  # Track number of batches for ce_loss
        self.embedding_running_mean = 1.0  # Initialize running mean for embedding_loss
        self.embedding_batch_count = 0  # Track number of batches for embedding_loss
        # Initialize cache using OrderedDict for LRU functionality
        self.embedding_cache = OrderedDict()
        self.cache_hits = 0
        # Existing initializations
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model
        self.embedder = self.model.embedder

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        ce_loss = outputs.loss  # Cross-entropy loss

        embedding_loss = torch.tensor(0.0, device=ce_loss.device)
        
        params = list(self.model.parameters()) + list(self.uncertainty_loss.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=1e-3)


        # Compute embedding loss at specified intervals
        logits = outputs.get("logits")
        pred_ids = torch.argmax(logits, dim=-1).detach().cpu()
        pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        pred_embeddings = []
        for text in pred_texts:
            if text in self.embedding_cache:
                pred_embedding = self.embedding_cache[text].to(ce_loss.device)
                self.cache_hits += 1
            else:
                pred_inputs = self.embedder_tokenizer(
                    [text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.embedder_tokenizer.model_max_length,
                ).to(ce_loss.device)
                pred_embedding = self.call_embedding_model(
                    input_ids=pred_inputs["input_ids"],
                    attention_mask=pred_inputs["attention_mask"],
                )
                self.embedding_cache[text] = pred_embedding.detach().cpu()
                if len(self.embedding_cache) > self.max_cache_size:
                    self.embedding_cache.popitem(last=False)
            pred_embeddings.append(pred_embedding)

        pred_embeddings = torch.stack(pred_embeddings)
        target_embeddings = inputs["frozen_embeddings"]
        if pred_embeddings.shape != target_embeddings.shape:
            pred_embeddings = pred_embeddings.view_as(target_embeddings)
        embedding_loss = 1 - nn.functional.cosine_similarity(pred_embeddings, target_embeddings, dim=-1).mean()

        # Update decaying window means
        self.ce_batch_count += 1
        self.ce_running_mean = (self.ce_running_mean * (self.ce_batch_count - 1) + ce_loss.item()) / self.ce_batch_count

        self.embedding_batch_count += 1
        self.embedding_running_mean = (self.embedding_running_mean * (self.embedding_batch_count - 1) + embedding_loss.item()) / self.embedding_batch_count

        # Normalize losses using running means
        normalized_ce_loss = ce_loss / self.ce_running_mean
        normalized_embedding_loss = embedding_loss / self.embedding_running_mean

        # Use the uncertainty loss function to compute the total loss
        total_loss, loss_info = self.uncertainty_loss(normalized_ce_loss, normalized_embedding_loss)

        # Log metrics
        self.log({
            'ce_loss': ce_loss.detach().item(),
            'embedding_loss': embedding_loss.detach().item(),
            'ce_running_mean': self.ce_running_mean,
            'embedding_running_mean': self.embedding_running_mean,
            'normalized_ce_loss': normalized_ce_loss.detach().item(),
            'normalized_embedding_loss': normalized_embedding_loss.detach().item(),
            'total_loss': total_loss.detach().item(),
            **loss_info,
            'cache_hits': self.cache_hits
        })

        return (total_loss, outputs) if return_outputs else total_loss

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Performs a training step. we override to compute data-specific metrics.
        """
        # TODO: Log training metrics from below... (How to do with huggingface?)
        self._compute_data_metrics(inputs=inputs)
        # self.log({ f"train/{k}": v for k,v in metrics.items() })
        return super().training_step(model, inputs)

    def evaluation_loop(
        self, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.

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
        """Edit keys posthumously on model load."""
        # Rename keys for backward compatibility w/ model trained before
        # we added extra dropout to the model
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
