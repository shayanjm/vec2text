import copy
import logging
from typing import Dict, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class RefinementBlock(nn.Module):
    """
    A small sub-network that refines a previously generated text. 
    We combine (frozen embedding + partial/hypothesis embedding + partial tokens)
    to produce improved tokens.

    In this example, we use a separate seq2seq sub-model, but you can simplify 
    or adapt to your use-case.
    """

    def __init__(self, config: InversionConfig):
        super().__init__()
        self.config = config

        # A small seq2seq sub-model for refinement. e.g., T5 or BART
        self.encoder_decoder = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path
        )

        # We'll assume a default dimension
        embedder_dim = 1536 if config.embedder_model_api else 768
        self.embedding_transform = nn.Sequential(
            nn.Linear(embedder_dim, embedder_dim),
            nn.GELU(),
            nn.Linear(embedder_dim, self.encoder_decoder.config.hidden_size),
        )
        self.layernorm = nn.LayerNorm(self.encoder_decoder.config.hidden_size)

    def forward_generate(
        self,
        frozen_embeddings: torch.Tensor,
        partial_embeddings: torch.Tensor,
        partial_ids: torch.Tensor,
        partial_mask: torch.Tensor,
        generation_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        We do a simple approach: combine the two embeddings, prepend them as special tokens, 
        then call self.encoder_decoder.generate() to produce refined text.
        """
        B = frozen_embeddings.size(0)
        device = frozen_embeddings.device

        # Transform them
        fe = self.embedding_transform(frozen_embeddings)
        pe = self.embedding_transform(partial_embeddings)
        combined = self.layernorm(fe + pe).unsqueeze(1)  # shape [B,1, hidden_size]

        # We'll embed the partial input tokens
        input_embs = self.encoder_decoder.get_encoder().embed_tokens(partial_ids)
        # shape: [B, seq_len, hidden_size]

        # Prepend 1 "combined" token
        new_inputs_embs = torch.cat([combined, input_embs], dim=1)
        # Build new attention mask
        extra_ones = torch.ones((B, 1), dtype=partial_mask.dtype, device=device)
        new_attn_mask = torch.cat([extra_ones, partial_mask], dim=1)

        refine_ids = self.encoder_decoder.generate(
            inputs_embeds=new_inputs_embs,
            attention_mask=new_attn_mask,
            **generation_kwargs
        )
        return refine_ids


class InversionModel(transformers.PreTrainedModel):
    """
    Overwrites your existing InversionModel with integrated iterative-refinement logic.
    This model:
      1) does a baseline "embedding -> text" decode,
      2) optionally repeats to refine the text (sub-block "RefinementBlock").
    """

    config_class = InversionConfig
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    tokenizer: transformers.PreTrainedTokenizer
    embedding_transform: nn.Module

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        embedder_model_api = config.embedder_model_api
        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input
        encoder_dropout_disabled = config.encoder_dropout_disabled
        decoder_dropout_disabled = config.decoder_dropout_disabled
        embeddings_from_layer_n = config.embeddings_from_layer_n

        # 1) Load your underlying seq2seq model
        encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )

        # 2) Load the embedder model + tokenizer
        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name,
            torch_dtype=config.embedder_torch_dtype
        )

        # 3) Load your main tokenizer
        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )

        num_repeat_tokens = config.num_repeat_tokens
        embedder_no_grad = config.embedder_no_grad

        self.encoder_decoder = encoder_decoder
        self.num_repeat_tokens = num_repeat_tokens

        # Decide dims
        if embedder_model_api:
            self.embedder_dim = 1536
            bottleneck_dim = self.embedder_dim
        elif isinstance(embedder, SentenceTransformer):
            self.embedder_dim = embedder.get_sentence_embedding_dimension()
            bottleneck_dim = self.embedder_dim
        else:
            self.embedder_dim = embedder.config.hidden_size
            bottleneck_dim = self.embedder_dim

        self.embedder_no_grad = embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.bottleneck_dim = bottleneck_dim

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        )

        if encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)

        self.tokenizer = tokenizer
        self.embedder = embedder
        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        self.embedder_fake_with_zeros = embedder_fake_with_zeros
        self.embedding_transform_strategy = "repeat"
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = vars(config).get("embedder_gaussian_noise_level", 0.0)

        if self.embedder_no_grad:
            for param in self.embedder.parameters():
                param.requires_grad = False
            self.embedder.eval()

        # (New) a refinement sub-block
        self.refinement_block = RefinementBlock(config)

    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
            return outputs.pooler_output
        else:
            if self.embeddings_from_layer_n is not None:
                assert hasattr(outputs, "hidden_states"), (
                    "output missing hidden states - set output_hidden_states=True?"
                )
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
        if self.embedder_no_grad:
            self.embedder.eval()

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
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert embeddings.ndim == 2, "frozen_embeddings must be [B, emb_dim]"
        else:
            # We must embed from scratch
            if self.embedder_no_grad:
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

        # "repeat" strategy
        if embeddings.dtype != self.dtype:
            embeddings = embeddings.to(self.dtype)
        repeated_embeddings = self.embedding_transform(embeddings)
        B = repeated_embeddings.shape[0]
        repeated_embeddings = repeated_embeddings.reshape(B, self.num_repeat_tokens, -1)
        inputs_embeds = repeated_embeddings
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        return inputs_embeds, attention_mask

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
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

    # ---------------------------------------------------------------------
    # 1) A helper to do the single-pass base decode
    # ---------------------------------------------------------------------
    def _base_generate(
        self, 
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        The old single-pass approach: embed => project => self.encoder_decoder.generate().
        """
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
        return self.encoder_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs
        )

    def _embed_partial_decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Re-embed a partial or final decode (text) into the same embedding space 
        as 'frozen_embeddings' for comparison or input to refinement.
        """
        text_strs = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        device = token_ids.device
        emb_inputs = self.embedder_tokenizer(
            text_strs, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            partial_emb = self.call_embedding_model(
                input_ids=emb_inputs["input_ids"],
                attention_mask=emb_inputs["attention_mask"],
            )
        return partial_emb

    # ---------------------------------------------------------------------
    # 2) The final .generate() with optional multi-step refinement
    # ---------------------------------------------------------------------
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any],
        refine_steps: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Overridden generate that can do:
          1) Single pass base decode
          2) If refine_steps>0, repeatedly feed partial decode into self.refinement_block
        """
        # 1) do normal single pass
        base_output = self._base_generate(inputs, generation_kwargs)
        if refine_steps <= 0:
            return base_output

        # 2) Iteratively refine
        current_ids = base_output
        device = current_ids.device

        # We'll assume we have "frozen_embeddings" in the inputs 
        # so the refinement block can use it as "target"
        frozen_embeddings = inputs["frozen_embeddings"].to(device)

        for step_idx in range(refine_steps):
            # Re-embed the partial decode
            partial_emb = self._embed_partial_decode(current_ids)

            # The partial decode tokens themselves:
            # might need to ensure we don't pass pad tokens or huge sequences
            # We'll do a minimal approach:
            partial_mask = (current_ids != self.encoder_decoder.config.pad_token_id).long()

            refined_ids = self.refinement_block.forward_generate(
                frozen_embeddings=frozen_embeddings,
                partial_embeddings=partial_emb,
                partial_ids=current_ids,
                partial_mask=partial_mask,
                generation_kwargs=generation_kwargs,
            )
            current_ids = refined_ids

        return current_ids
