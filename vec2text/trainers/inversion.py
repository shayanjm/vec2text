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
        # Forward pass with labels to compute cross-entropy loss internally
        outputs = model(**inputs)
        ce_loss = outputs.loss  # Cross-entropy loss computed by the model

        # Get logits for generating predicted tokens
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

        # Get Predicted Tokens
        pred_ids = torch.argmax(logits, dim=-1)  # Shape: (batch_size, sequence_length)

        # Move pred_ids to CPU and convert to list
        pred_ids_cpu = pred_ids.detach().cpu().tolist()

        # Decode Predicted Tokens to Text
        pred_texts = self.tokenizer.batch_decode(pred_ids_cpu, skip_special_tokens=True)

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

        # Total Loss: Combine CE loss and embedding loss
        total_loss = ce_loss + self.embedding_loss_weight * embedding_loss

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
