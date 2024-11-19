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
        ce_loss,
        cosine_embedding_loss,
    ):
        # Compute uncertainties
        sigma_ce = torch.exp(log_sigma_ce)
        sigma_cosine_embedding = torch.exp(log_sigma_cosine_embedding)

        # Weighted total loss
        weighted_ce_loss = ce_loss / (2 * sigma_ce**2) + log_sigma_ce
        weighted_cosine_embedding_loss = (
            cosine_embedding_loss / (2 * sigma_cosine_embedding**2)
            + log_sigma_cosine_embedding
        )

        # Combine losses
        total_loss = weighted_ce_loss + weighted_cosine_embedding_loss

        return total_loss, {
            "weighted_ce_loss": weighted_ce_loss.detach().item(),
            "weighted_cosine_embedding_loss": weighted_cosine_embedding_loss.detach().item(),
            "log_sigma_ce": log_sigma_ce.detach().item(),
            "log_sigma_cosine_embedding": log_sigma_cosine_embedding.detach().item(),
        }


class InversionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding_loss_count = 0  # Track number of accumulated steps
        self.ce_running_mean = 1.0  # Initialize running mean for ce_loss
        self.ce_batch_count = 0  # Track number of batches for ce_loss
        self.cosine_embedding_running_mean = (
            1.0  # Initialize running mean for cosine_embedding_loss
        )
        self.cosine_embedding_batch_count = (
            0  # Track number of batches for cosine_embedding_loss
        )
        # Existing initializations
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model
        self.embedder = self.model.embedder

    def compute_loss(self, model, inputs, return_outputs=False):
        device = inputs["input_ids"].device
        batch_size = inputs["input_ids"].size(0)
        target_embeddings = inputs[
            "frozen_embeddings"
        ]  # Shape: (batch_size, embedding_dim)

        # Set model to evaluation mode for generation
        model.eval()

        with torch.no_grad():
            # Generate sequences with sampling
            generation_kwargs = {
                "max_length": self.model.config.max_length,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                "num_return_sequences": 1,
            }
            generated_ids = self.model.generate(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
            )

        # Set model back to training mode
        model.train()

        # Decode generated sequences
        pred_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Compute rewards
        rewards = []
        for i in range(batch_size):
            pred_text = pred_texts[i]
            target_embedding = target_embeddings[i]

            # Encode predicted text using embedder_tokenizer
            pred_input = self.embedder_tokenizer(
                pred_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.embedder_tokenizer.model_max_length,
            ).to(device)

            # Call the embedding model on the text
            pred_embedding = self.call_embedding_model(
                input_ids=pred_input["input_ids"],
                attention_mask=pred_input["attention_mask"],
            )

            # Compute cosine similarity
            cosine_sim = nn.functional.cosine_similarity(
                pred_embedding, target_embedding.unsqueeze(0), dim=-1
            ).mean()
            rewards.append(cosine_sim.item())

        rewards = torch.tensor(rewards).to(device)

        # Prepare labels and decoder_input_ids
        labels = generated_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask padding tokens

        # Shift labels to create decoder_input_ids
        decoder_input_ids = self.model.encoder_decoder._shift_right(labels)

        # Ensure decoder_input_ids and labels have the same length
        max_seq_length = labels.size(1)
        decoder_input_ids = decoder_input_ids[:, :max_seq_length]

        # Create decoder_attention_mask
        decoder_attention_mask = (
            decoder_input_ids != self.tokenizer.pad_token_id
        ).long()

        # Call the model to get outputs
        outputs = model(
            **inputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        # Compute log probabilities
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

        # Ensure logits and labels have the same sequence length
        seq_length = labels.size(1)
        logits = logits[:, :seq_length, :]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute per-token losses
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Create loss mask
        loss_mask = (shift_labels.view(-1) != -100).float()

        # Apply the loss mask
        loss = loss * loss_mask

        # Sum over valid tokens
        log_probs = -loss.sum(dim=0) / loss_mask.sum(dim=0)

        # Compute policy gradient loss
        baseline = rewards.mean()
        advantages = rewards - baseline

        loss = -(advantages * log_probs).mean()

        # Log metrics
        self.log(
            {
                "average_reward": rewards.mean().item(),
                "policy_loss": loss.detach().item(),
            }
        )

        return (loss, outputs) if return_outputs else loss

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
