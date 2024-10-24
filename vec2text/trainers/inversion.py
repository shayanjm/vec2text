import math
from typing import Dict

import torch
import torch.nn as nn
import transformers

from vec2text.trainers.base import BaseTrainer


class InversionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ######################################################
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model
        self.embedder = self.model.embedder

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute Typical Cross-Entropy Loss
        labels = inputs.get("labels")
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Get Predicted Tokens
        pred_ids = torch.argmax(logits, dim=-1)

        # Debugging: Check for invalid token IDs
        vocab_size = self.tokenizer.vocab_size
        if torch.any(pred_ids < 0) or torch.any(pred_ids >= vocab_size):
            print(f"Invalid token IDs detected. Min ID: {pred_ids.min()}, Max ID: {pred_ids.max()}, Vocab Size: {vocab_size}")

        pred_ids = pred_ids.detach().cpu()

        # Decode Predicted Tokens to Text
        pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # Tokenize predicted texts using embedder_tokenizer
        pred_inputs = self.embedder_tokenizer(
            pred_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.embedder_tokenizer.model_max_length,
        ).to(logits.device)

        # Use call_embedding_model to get embeddings of predicted texts
        pred_embeddings = self.call_embedding_model(
            input_ids=pred_inputs['input_ids'],
            attention_mask=pred_inputs['attention_mask'],
        )

        # Get Target Embeddings
        target_embeddings = inputs['frozen_embeddings']

        # Compute Embedding Distance (e.g., Mean Squared Error)
        embedding_loss = nn.functional.mse_loss(pred_embeddings, target_embeddings)

        # Access log_var_ce and log_var_embedding via model.module if necessary
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            log_var_ce = model.module.log_var_ce
            log_var_embedding = model.module.log_var_embedding
        else:
            log_var_ce = model.log_var_ce
            log_var_embedding = model.log_var_embedding

        # Compute total loss with uncertainty weighting
        precision_ce = torch.exp(-log_var_ce)
        precision_embedding = torch.exp(-log_var_embedding)

        total_loss = (precision_ce * ce_loss + log_var_ce) + \
                        (precision_embedding * embedding_loss + log_var_embedding)

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
