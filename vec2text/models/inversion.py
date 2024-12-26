import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer

from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import (
    FREEZE_STRATEGIES,
    disable_dropout,
    freeze_params,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    load_tokenizer,
    mean_pool,
)
from vec2text.utils import embed_api

logger = logging.getLogger(__name__)

# IMPORTANT:
# Instead of huggingface's AutoModelForSeq2SeqLM, we will use TRL's 
# AutoModelForSeq2SeqLMWithValueHead for RL training.
from trl import AutoModelForSeq2SeqLMWithValueHead


class InversionModel(nn.Module):
    """
    This version simply wraps a T5-small with a value head for RL,
    plus your embedding logic for conditioning and reward computations.
    """

    def __init__(self, config: InversionConfig):
        super().__init__()

        self.config = config
        embedder_model_api = config.embedder_model_api
        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input
        encoder_dropout_disabled = config.encoder_dropout_disabled
        decoder_dropout_disabled = config.decoder_dropout_disabled
        embeddings_from_layer_n = config.embeddings_from_layer_n

        # Load T5 model with a value head from TRL
        # (Under the hood, it's basically T5ForConditionalGeneration + value tower.)
        self.encoder_decoder = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
            config.model_name_or_path
        )

        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        )

        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )
        self.tokenizer = tokenizer

        self.num_repeat_tokens = config.num_repeat_tokens
        self.embedder_no_grad = config.embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input

        # Possibly freeze dropout, etc.
        if encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)

        # store embedder
        self.embedder = embedder
        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        self.embedder_fake_with_zeros = embedder_fake_with_zeros
        self.embeddings_from_layer_n = embeddings_from_layer_n

        # Determine embedder dim -> set up linear transformation
        if self.embedder_model_api:
            # Hard-code OpenAI embedding dim, for example
            self.embedder_dim = 1536
        elif isinstance(embedder, SentenceTransformer):
            self.embedder_dim = embedder.get_sentence_embedding_dimension()
        else:
            self.embedder_dim = embedder.config.hidden_size

        # If we want a distinct transform from embedding -> T5 hidden states:
        encoder_hidden_dim = self.encoder_decoder.config.d_model
        bottleneck_dim = self.embedder_dim  # or something else
        self.bottleneck_dim = bottleneck_dim

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim * self.num_repeat_tokens),
        )

        # freeze embedder if needed
        if self.embedder_no_grad:
            for param in self.embedder.parameters():
                param.requires_grad = False
            self.embedder.eval()

        self.log_sigma_ce = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cosine_embedding = nn.Parameter(torch.tensor(0.0))

        self.noise_level = vars(config).get("embedder_gaussian_noise_level", 0.0)

    def freeze(self, freeze_strategy: str):
        # example from your code
        assert freeze_strategy in FREEZE_STRATEGIES
        if freeze_strategy == "decoder":
            self._freeze_decoder()
        elif freeze_strategy == "encoder":
            self._freeze_encoder()
        elif freeze_strategy == "encoder_and_decoder":
            self._freeze_encoder()
            self._freeze_decoder()
            freeze_params(self.encoder_decoder.shared)
        elif freeze_strategy == "none":
            pass
        else:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")

    def _freeze_encoder(self):
        freeze_params(self.encoder_decoder.encoder)

    def _freeze_decoder(self):
        freeze_params(self.encoder_decoder.decoder)
        freeze_params(self.encoder_decoder.lm_head)

    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            if self.embeddings_from_layer_n is not None:
                hidden_state = outputs.hidden_states[self.embeddings_from_layer_n]
                embeddings = mean_pool(hidden_state, attention_mask)
            else:
                hidden_state = outputs.last_hidden_state
                embeddings = mean_pool(hidden_state, attention_mask)
            return embeddings

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.embedder_fake_with_zeros:
            batch_size = input_ids.shape[0]
            return torch.zeros(
                (batch_size, self.embedder_dim),
                dtype=torch.float32,
                device=self.embedder_device,
            )
        elif self.embedder_model_api:
            embeddings = embed_api(
                input_ids=input_ids,
                embedder_tokenizer=self.embedder_tokenizer,
                api_name=self.embedder_model_api,
            )
        elif isinstance(self.embedder, SentenceTransformer):
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids
            model_output = self.embedder(model_inputs)
            embeddings = model_output["sentence_embedding"]
        else:
            model_output = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self._process_embedder_output(model_output, attention_mask)

        if self.training and self.noise_level > 0:
            embeddings += self.noise_level * torch.randn(
                embeddings.shape, device=embeddings.device
            )
        return embeddings

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert not ((embedder_input_ids is None) and (frozen_embeddings is None))
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2
        elif self.embedder_no_grad:
            with torch.no_grad():
                embeddings = self.call_embedding_model(
                    input_ids=embedder_input_ids,
                    attention_mask=embedder_attention_mask,
                )
        else:
            embeddings = self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )

        # transform into repeated embeddings
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.to(torch.float32)
        repeated_embeddings = self.embedding_transform(embeddings)
        embeddings = repeated_embeddings.reshape(
            (*repeated_embeddings.shape[:-1], self.num_repeat_tokens, -1)
        )
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )
        return embeddings, attention_mask

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
        return self.encoder_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    def forward(
        self,
        embedder_input_ids: torch.Tensor = None,
        embedder_attention_mask: torch.Tensor = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        By default, the forward pass just calls the underlying T5 with or without
        inputs_embeds, to keep a typical seq2seq forward signature.
        """
        if input_ids is not None and attention_mask is not None:
            # Use token IDs directly
            return self.encoder_decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
            )
        else:
            # Use embedder and project
            inputs_embeds, attention_mask = self.embed_and_project(
                embedder_input_ids=embedder_input_ids,
                embedder_attention_mask=embedder_attention_mask,
                frozen_embeddings=frozen_embeddings,
            )
            return self.encoder_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
            )