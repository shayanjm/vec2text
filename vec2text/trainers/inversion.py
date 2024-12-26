import math
from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import transformers

from trl import PPOConfig, PPOTrainer
from copy import deepcopy

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

        self.ppo_config = PPOConfig(
            model_name=self.model.config.model_name_or_path,
            learning_rate=self.args.learning_rate,  # or separate LR for PPO
            batch_size=self.args.per_device_train_batch_size,
            # Additional PPO hyperparams here...
        )

        self.ref_model = deepcopy(self.model.encoder_decoder).eval()
         # Create the PPOTrainer with your T5-value-head policy
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model.encoder_decoder,  # policy
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        # --------------------------------------------------------------------------------
        # TRL PPO logic
        # --------------------------------------------------------------------------------
        device = inputs["frozen_embeddings"].device
        batch_size = inputs["frozen_embeddings"].size(0)

        # 1) Generate using the current policy
        generation_kwargs = {
            "max_length": self.model.config.max_length,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
        }
        generated_ids = self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # 2) Compute reward as e.g. cosine similarity with target embeddings
        rewards = []
        for i in range(batch_size):
            pred_text = generated_texts[i]
            tgt_emb = inputs["frozen_embeddings"][i]

            # embed predicted text
            enc = self.embedder_tokenizer(
                pred_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_length,
            ).to(device)
            pred_emb = self.call_embedding_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            cos_sim = nn.functional.cosine_similarity(pred_emb, tgt_emb.unsqueeze(0), dim=-1)
            rewards.append(cos_sim.item())
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        # 3) PPO step => queries = input prompt, responses = generated text
        # For T5, the "query" is typically the encoder input. E.g.:
        queries = inputs["input_ids"]  # or "embedder_input_ids" if your main input is embedder-based
        responses = generated_ids

        # step returns a stats dict with PPO training info
        stats = self.ppo_trainer.step(queries, responses, rewards)

        # HF Trainer expects a scalar loss to backprop
        # TRL's .step() already does the PPO backprop update, so we can return 0
        # or return stats["ppo/total_loss"] if you prefer to log some numeric value
        ppo_loss = stats.get("objective/kl", 0.0)
        final_loss = torch.tensor(ppo_loss).float().to(device)

        # log stats
        self.log({
            "train/ppo_loss": ppo_loss,
            "train/reward_mean": rewards.mean().item(),
        })

        return (final_loss, None) if return_outputs else final_loss

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
        # Explicitly set model to evaluation mode
        self.model.eval()

        output = super().evaluation_loop(*args, **kwargs)

        metric_key_prefix = kwargs["metric_key_prefix"]
        try:
            perplexity = math.exp(output.metrics[f"{metric_key_prefix}_loss"])
        except KeyError:
            perplexity = -1
        except OverflowError:
            perplexity = float("inf")
        output.metrics[f"{metric_key_prefix}_perplexity"] = perplexity

        # Restore to training mode after evaluation
        self.model.train()

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
