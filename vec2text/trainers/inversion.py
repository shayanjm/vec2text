import math
from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import transformers

from vec2text.trainers.base import BaseTrainer


class InversionTrainer(BaseTrainer):
    def __init__(self, *args, max_cache_size=10000, embedding_loss_interval=100, **kwargs):
        super().__init__(*args, **kwargs)
        # New parameters: maximum cache size and embedding loss interval
        self.max_cache_size = max_cache_size
        self.embedding_loss_interval = embedding_loss_interval
        self.embedding_loss_accumulator = 0.0  # Initialize embedding loss accumulator
        self.embedding_loss_count = 0  # Track number of accumulated steps
        # Initialize cache using OrderedDict for LRU functionality
        self.embedding_cache = OrderedDict()
        # Existing initializations
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model
        self.embedder = self.model.embedder

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        ce_loss = outputs.loss  # Cross-entropy loss computed by the model

        # Access log_var_ce and log_var_embedding
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            log_var_ce = model.module.log_var_ce
            log_var_embedding = model.module.log_var_embedding
        else:
            log_var_ce = model.log_var_ce
            log_var_embedding = model.log_var_embedding

        precision_ce = torch.exp(-log_var_ce)
        total_loss = precision_ce * ce_loss + log_var_ce

        # Compute embedding loss only at intervals
        if (self.state.global_step + 1) % self.embedding_loss_interval == 0:
            logits = outputs.get("logits")  # Shape: (batch_size, sequence_length, vocab_size)
            pred_ids = torch.argmax(logits, dim=-1)
            pred_ids = pred_ids.detach().cpu()

            # Decode Predicted Tokens to Text
            pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            # Initialize list for embeddings
            embeddings_list = []
            for text in pred_texts:
                if text in self.embedding_cache:
                    # Retrieve from cache and move to device
                    pred_embedding = self.embedding_cache[text].to(ce_loss.device)
                else:
                    # Tokenize and compute embedding
                    pred_inputs = self.embedder_tokenizer(
                        [text],
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.embedder_tokenizer.model_max_length,
                    ).to(ce_loss.device)
                    pred_embedding = self.call_embedding_model(
                        input_ids=pred_inputs['input_ids'],
                        attention_mask=pred_inputs['attention_mask'],
                    )
                    # Store in cache and enforce max cache size
                    self.embedding_cache[text] = pred_embedding.detach().cpu()
                    if len(self.embedding_cache) > self.max_cache_size:
                        # Remove oldest item
                        self.embedding_cache.popitem(last=False)
                embeddings_list.append(pred_embedding)

            # Stack embeddings
            pred_embeddings = torch.cat(embeddings_list, dim=0)

            # Get Target Embeddings
            target_embeddings = inputs['frozen_embeddings']

            # Ensure the shapes match
            if pred_embeddings.shape != target_embeddings.shape:
                # Adjust shapes if necessary
                pred_embeddings = pred_embeddings.view_as(target_embeddings)

            # Compute Embedding Distance (e.g., Mean Squared Error)
            embedding_loss = nn.functional.mse_loss(pred_embeddings, target_embeddings)
            self.embedding_loss_accumulator += embedding_loss
            self.embedding_loss_count += 1

            # Compute average embedding loss and add to total loss
            avg_embedding_loss = self.embedding_loss_accumulator / self.embedding_loss_count
            precision_embedding = torch.exp(-log_var_embedding)
            total_loss += precision_embedding * avg_embedding_loss + log_var_embedding

            # Reset the accumulator and count
            self.embedding_loss_accumulator = 0.0
            self.embedding_loss_count = 0

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
