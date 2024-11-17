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

    def forward(
        self,
        log_sigma_ce,
        log_sigma_cosine_embedding,
        log_sigma_mse_embedding,
        ce_loss,
        cosine_embedding_loss,
        mse_embedding_loss,
    ):
        # Compute uncertainties
        sigma_ce = torch.exp(log_sigma_ce)
        sigma_cosine_embedding = torch.exp(log_sigma_cosine_embedding)
        sigma_mse_embedding = torch.exp(log_sigma_mse_embedding)

        # Weighted total loss
        weighted_ce_loss = ce_loss / (2 * sigma_ce**2) + log_sigma_ce
        weighted_cosine_embedding_loss = (
            cosine_embedding_loss / (2 * sigma_cosine_embedding**2)
            + log_sigma_cosine_embedding
        )
        weighted_mse_embedding_loss = (
            mse_embedding_loss / (2 * sigma_mse_embedding**2) + log_sigma_mse_embedding
        )

        # Combine losses
        total_loss = (
            weighted_ce_loss
            + weighted_cosine_embedding_loss
            + weighted_mse_embedding_loss
        )

        return total_loss, {
            "weighted_ce_loss": weighted_ce_loss.detach().item(),
            "weighted_cosine_embedding_loss": weighted_cosine_embedding_loss.detach().item(),
            "weighted_mse_embedding_loss": weighted_mse_embedding_loss.detach().item(),
            "log_sigma_ce": log_sigma_ce.detach().item(),
            "log_sigma_cosine_embedding": log_sigma_cosine_embedding.detach().item(),
            "log_sigma_mse_embedding": log_sigma_mse_embedding.detach().item(),
        }


class InversionTrainer(BaseTrainer):
    def __init__(self, *args, max_cache_size=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_loss = UncertaintyLoss()

        self.max_cache_size = max_cache_size
        self.embedding_loss_count = 0  # Track number of accumulated steps
        self.ce_running_mean = 1.0  # Initialize running mean for ce_loss
        self.ce_batch_count = 0  # Track number of batches for ce_loss
        self.cosine_embedding_running_mean = (
            1.0  # Initialize running mean for cosine_embedding_loss
        )
        self.cosine_embedding_batch_count = (
            0  # Track number of batches for cosine_embedding_loss
        )
        self.mse_embedding_running_mean = (
            1.0  # Initialize running mean for mse_embedding_loss
        )
        self.mse_embedding_batch_count = (
            0  # Track number of batches for mse_embedding_loss
        )
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

        cosine_embedding_loss = torch.tensor(0.0, device=ce_loss.device)
        mse_embedding_loss = torch.tensor(0.0, device=ce_loss.device)

        params = list(self.model.parameters())
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

        cosine_embedding_loss = (
            1
            - nn.functional.cosine_similarity(
                pred_embeddings, target_embeddings, dim=-1
            ).mean()
        )
        mse_embedding_loss = nn.functional.mse_loss(pred_embeddings, target_embeddings)

        # Update decaying window means
        self.ce_batch_count += 1
        self.ce_running_mean = (
            self.ce_running_mean * (self.ce_batch_count - 1) + ce_loss.item()
        ) / self.ce_batch_count

        self.cosine_embedding_batch_count += 1
        self.cosine_embedding_running_mean = (
            self.cosine_embedding_running_mean * (self.cosine_embedding_batch_count - 1)
            + cosine_embedding_loss.item()
        ) / self.cosine_embedding_batch_count

        self.mse_embedding_batch_count += 1
        self.mse_embedding_running_mean = (
            self.mse_embedding_running_mean * (self.mse_embedding_batch_count - 1)
            + mse_embedding_loss.item()
        ) / self.mse_embedding_batch_count

        # Normalize losses using running means
        normalized_ce_loss = ce_loss / self.ce_running_mean
        normalized_cosine_embedding_loss = (
            cosine_embedding_loss / self.cosine_embedding_running_mean
        )
        normalized_mse_embedding_loss = (
            mse_embedding_loss / self.mse_embedding_running_mean
        )

        # Use the uncertainty loss function to compute the total loss
        total_loss, loss_info = self.uncertainty_loss(
            self.model.log_sigma_ce,
            self.model.log_sigma_cosine_embedding,
            self.model.log_sigma_mse_embedding,
            normalized_ce_loss,
            normalized_cosine_embedding_loss,
            normalized_mse_embedding_loss,
        )

        # Log metrics
        self.log(
            {
                "ce_loss": ce_loss.detach().item(),
                "cosine_embedding_loss": cosine_embedding_loss.detach().item(),
                "mse_embedding_loss": mse_embedding_loss.detach().item(),
                "ce_running_mean": self.ce_running_mean,
                "cosine_embedding_running_mean": self.cosine_embedding_running_mean,
                "mse_embedding_running_mean": self.mse_embedding_running_mean,
                "normalized_ce_loss": normalized_ce_loss.detach().item(),
                "normalized_cosine_embedding_loss": normalized_cosine_embedding_loss.detach().item(),
                "normalized_mse_embedding_loss": normalized_mse_embedding_loss.detach().item(),
                "total_loss": total_loss.detach().item(),
                **loss_info,
                "cache_hits": self.cache_hits,
            }
        )

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
