import math
from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, supervised_training: bool = False, *args, **kwargs):
        """
        :param supervised_training: If True, the trainer will do a 
            supervised cross-entropy training using 'labels' as 
            ground-truth text. Otherwise, RL-based training is used.
        """
        super().__init__(*args, **kwargs)

        self.supervised_training = supervised_training

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
        """
        The main training loop. Depending on `self.supervised_training`:
          - Supervised mode (cross-entropy on known text).
          - RL mode (policy gradient with cosine-sim reward).
        """
        device = inputs["frozen_embeddings"].device
        batch_size = inputs["frozen_embeddings"].size(0)

        # --------------------------------------------------------------------
        # 1) Supervised Training Mode (if you already have ground-truth text)
        # --------------------------------------------------------------------
        if self.supervised_training:
            # We expect `inputs["labels"]` to contain the ground-truth text token IDs.
            # Force the model to generate from `frozen_embeddings` → decode text → compare with labels.
            outputs = model(
                frozen_embeddings=inputs["frozen_embeddings"],
                labels=inputs["labels"],  # teacher-forcing cross-entropy
            )
            loss = outputs.loss  # HF integrates CE loss for Seq2Seq models

            # Optionally log something
            self.log({"supervised_ce_loss": loss.item()})
            return (loss, outputs) if return_outputs else loss

        # -----------------------------------------------------
        # 2) RL Mode: sample text, compute reward, do PG update
        # -----------------------------------------------------

        # Put model in eval mode while sampling
        model.eval()

        # Lower-entropy sampling to avoid extremely random text
        generation_kwargs = {
            "max_length": self.model.config.max_length,
            "do_sample": True,
            "top_k": 20,         # lowered from 50
            "top_p": 0.90,       # lowered from 0.95
            "temperature": 0.8,  # new
            "num_return_sequences": 1,
        }
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
            )

        # Switch back to train mode
        model.train()

        # Decode sequences
        pred_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Compute rewards as cosine similarity
        target_embeddings = inputs["frozen_embeddings"]  # shape: (batch, emb_dim)
        rewards = []
        for i in range(batch_size):
            pred_text = pred_texts[i]
            # Re-embed predicted text
            pred_input = self.embedder_tokenizer(
                pred_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_length,
            ).to(device)

            pred_embedding = self.call_embedding_model(
                input_ids=pred_input["input_ids"],
                attention_mask=pred_input["attention_mask"],
            )  # shape ~ (1, emb_dim)

            cos_sim = F.cosine_similarity(
                pred_embedding, target_embeddings[i].unsqueeze(0), dim=-1
            ).mean()
            rewards.append(cos_sim.item())

        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Prepare labels for policy gradient
        labels = generated_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # (Optional) pad or truncate
        max_len = self.model.config.max_length
        if generated_ids.size(1) < max_len:
            generated_ids = nn.functional.pad(
                generated_ids,
                (0, max_len - generated_ids.size(1)),
                value=self.tokenizer.pad_token_id,
            )
        else:
            generated_ids = generated_ids[:, :max_len]

        if labels.size(1) < max_len:
            labels = nn.functional.pad(
                labels,
                (0, max_len - labels.size(1)),
                value=-100,
            )
        else:
            labels = labels[:, :max_len]

        # Shift labels to create decoder_input_ids
        decoder_input_ids = self.model.encoder_decoder._shift_right(labels)

        # Create decoder_attention_mask
        # (Though T5 often doesn't strictly require it, we preserve your code.)
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id).long()

        # Forward pass with `frozen_embeddings` only
        outputs = model(
            frozen_embeddings=inputs["frozen_embeddings"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        # Compute negative log_probs for PG
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        logits = outputs.logits  # (batch_size, seq_length, vocab_size)

        # Ensure logits and labels have the same seq_length
        seq_length = labels.size(1)
        logits = logits[:, :seq_length, :]

        shift_logits = logits[:, :-1, :].contiguous()   # (batch, seq_length-1, vocab)
        shift_labels = labels[:, 1:].contiguous()       # (batch, seq_length-1)

        # Per-token CE
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )  # shape: (batch*(seq_length -1))

        # Mask out -100
        loss_mask = (shift_labels.view(-1) != -100).float()
        per_token_loss = per_token_loss * loss_mask

        # Reshape
        per_token_loss = per_token_loss.view(batch_size, -1)
        loss_mask = loss_mask.view(batch_size, -1)

        # Sum over valid tokens => total CE per sequence => negative for log_prob
        log_probs = -(per_token_loss.sum(dim=1) / (loss_mask.sum(dim=1) + 1e-8))

        # Policy gradient step with mean-baseline
        baseline = rewards.mean()
        advantages = rewards - baseline

        # Advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Final policy gradient loss
        loss = -(advantages * log_probs).mean()

        # Log metrics
        self.log({
            "average_reward": rewards.mean().item(),
            "policy_loss": loss.detach().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "log_probs_mean": log_probs.mean().item(),
        })

        return (loss, outputs) if return_outputs else loss

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        """
        Generation call used by HF Trainer if we do e.g. trainer.predict(...).
        Pass `frozen_embeddings` in `inputs`, and define `generation_kwargs`.
        """
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs a training step. We override to compute data-specific metrics
        before the loss is computed, if needed.
        """
        self._compute_data_metrics(inputs=inputs)
        return super().training_step(model, inputs)

    def evaluation_loop(self, *args, **kwargs) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics. We add perplexity as well.
        """
        # Explicitly set model to evaluation mode
        self.model.eval()

        output = super().evaluation_loop(*args, **kwargs)

        metric_key_prefix = kwargs["metric_key_prefix"]
        try:
            ppl = math.exp(output.metrics[f"{metric_key_prefix}_loss"])
        except KeyError:
            ppl = -1
        except OverflowError:
            ppl = float("inf")

        output.metrics[f"{metric_key_prefix}_perplexity"] = ppl

        # Restore training mode
        self.model.train()

        return output

    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """
        Edit keys posthumously on model load for backward compatibility
        if older checkpoints had a different layer indexing.
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
