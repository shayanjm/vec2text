import copy
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


class InversionModel(transformers.PreTrainedModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """

    config_class = InversionConfig
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer  # embedder's tokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool  # Whether to use LoRA for the encoder-decoder model
    tokenizer: transformers.PreTrainedTokenizer  # encoder_decoder's tokenizer
    embedding_transform: nn.Module  # Module that transformers embedder output into encoder-decoder input
    bottleneck_dim: int  # Bottleneck dimension for embedding_transform
    num_repeat_tokens: int  # Sequence length for repeating embedder embedding for encoder-decoder input
    embedder_dim: int  # Hidden dimension of embedding model
    embedder_no_grad: bool  # Disable gradients for embedding model
    embedder_fake_with_zeros: bool  # Whether to just provide zeros as input for encoder-decoder (unconditional)
    embedding_transform_strategy: str  # Way to transform bottleneck embedding into input for encoder-decoder
    use_frozen_embeddings_as_input: bool  # Whether to train/evaluate on frozen embeddings
    embedded_tokens: torch.Tensor  # used for decoding
    embedder_model_api: Optional[str]
    
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        embedder_model_api = config.embedder_model_api
        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input
        encoder_dropout_disabled = config.encoder_dropout_disabled
        decoder_dropout_disabled = config.decoder_dropout_disabled
        embeddings_from_layer_n = config.embeddings_from_layer_n

        encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )

        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        )

        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )
        num_repeat_tokens = config.num_repeat_tokens
        embedder_no_grad = config.embedder_no_grad

        self.encoder_decoder = encoder_decoder  # .to_bettertransformer()
        ######################################################
        self.num_repeat_tokens = num_repeat_tokens

        self.embedder_is_decoder = False

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        if embedder_model_api:
            assert use_frozen_embeddings_as_input, "must precompute embeddings w/ api"
            # Hard-code OpenAI embedding dim
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

        if encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)
        ######################################################
        self.tokenizer = tokenizer
        self.embedder = embedder
        if self.embedder_no_grad:
            for param in self.embedder.parameters():
                param.requires_grad = False

            self.embedder.eval()

        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        # self.freeze(freeze_strategy=config.freeze_strategy)
        self.embedder_fake_with_zeros = embedder_fake_with_zeros

        self.embedding_transform_strategy = config.embedding_transform_strategy
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = vars(config).get("embedder_gaussian_noise_level")

        if self.embedding_transform_strategy == "overlap_chunking":
            # For chunk-based approach
            self.embedding_transform = nn.Sequential(
                nn.Linear(self.embedder_dim, encoder_hidden_dim),
                nn.Dropout(self.encoder_decoder.config.dropout_rate),
                nn.GELU(),
                nn.LayerNorm(encoder_hidden_dim),
            )
        else:
            # Fallback to the old "repeat" bridging
            self.embedding_transform = nn.Sequential(
                nn.Linear(self.embedder_dim, bottleneck_dim),
                nn.Dropout(self.encoder_decoder.config.dropout_rate),
                nn.GELU(),
                nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
            )

    def _freeze_encoder(self):
        freeze_params(self.encoder_decoder.encoder)

    def _freeze_decoder(self):
        # github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L1229-L1231
        freeze_params(self.encoder_decoder.decoder)
        freeze_params(self.encoder_decoder.lm_head)

    def _chunk_and_embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Splits each sequence in (input_ids, attention_mask) into overlapping chunks,
        calls the embedding model on each chunk, and returns a
        (batch_size, num_chunks, embedder_dim) tensor of chunk embeddings.
        """
        batch_size, full_seq_len = input_ids.shape
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap

        all_chunk_embeddings = []
        start = 0
        while start < full_seq_len:
            end = start + chunk_size
            if end > full_seq_len:
                end = full_seq_len

            # Slice out chunk
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end]

            # Call your existing code to embed
            chunk_embedding = self.call_embedding_model(
                input_ids=chunk_ids, attention_mask=chunk_mask
            )
            # shape => (batch_size, embedder_dim)
            all_chunk_embeddings.append(chunk_embedding.unsqueeze(1))

            # Move forward by chunk_size - overlap
            step_size = max(1, chunk_size - overlap)
            start += step_size

            if start >= full_seq_len:
                break

        if len(all_chunk_embeddings) == 0:
            return torch.zeros(
                (batch_size, 1, self.embedder_dim),
                dtype=torch.float32,
                device=self.embedder_device,
            )

        return torch.cat(all_chunk_embeddings, dim=1)  # => (batch_size, num_chunks, embedder_dim)

    def freeze(self, freeze_strategy: str):
        assert freeze_strategy in FREEZE_STRATEGIES

        if freeze_strategy == "decoder":
            self._freeze_decoder()
        elif freeze_strategy == "encoder":
            self._freeze_encoder()
        elif freeze_strategy == "encoder_and_decoder":
            self._freeze_encoder()
            self._freeze_decoder()
            # in this case, freeze embeddings too
            freeze_params(self.encoder_decoder.shared)
        elif freeze_strategy == "none":
            pass
        else:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")

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
                assert hasattr(
                    outputs, "hidden_states"
                ), "output missing hidden states - did you remember to initialize the model with output_hidden_states=True?"
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
        # token_type_ids: Optional[torch.Tensor] = None, # not used
    ) -> torch.Tensor:
        embedder = self.embedder
        # print("** call_embedding_model")
        if self.embedder_no_grad:
            embedder.eval()

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
            # sentence-transformers is kind of really annoying
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids
            model_output = embedder(model_inputs)
            embeddings = model_output["sentence_embedding"]
        else:
            model_output = embedder(input_ids=input_ids, attention_mask=attention_mask)
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
        """
        Produces T5 encoder inputs: (batch_size, seq_len, hidden_dim)
        plus an attention mask of shape (batch_size, seq_len).

        If 'frozen_embeddings' is provided, we skip calling the embedder and just project
        those embeddings. Otherwise, we either:
        1) do overlap chunking (if self.embedding_transform_strategy == 'overlap_chunking'),
        2) do repeat-based strategy (if 'repeat'),
        3) or raise an error if unknown.
        """

        # -------------------------------------------------------------------------
        # 1) If we have frozen embeddings, we assume they've either been pre-chunked
        #    or are single vectors. We handle them directly and skip calling embedder.
        # -------------------------------------------------------------------------
        if frozen_embeddings is not None:
            # e.g. shape might be (batch_size, embed_dim) or (batch_size, num_chunks, embed_dim).
            embeddings = frozen_embeddings
            # If you only stored single-vector embeddings, that means it's shape (B, embed_dim).
            # If you stored them chunked, you'd have (B, num_chunks, embed_dim).
            # We'll handle the dimension logic inside the strategy check below.
        else:
            # Actually embed from input_ids
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

        # Make sure we're matching the model's dtype, if needed
        if embeddings.dtype != self.dtype:
            embeddings = embeddings.to(self.dtype)

        # -------------------------------------------------------------------------
        # 2) Transform the embeddings into a sequence that T5 can attend over
        # -------------------------------------------------------------------------
        if self.embedding_transform_strategy == "overlap_chunking":
            # (A) Overlap-chunking approach
            # If we have 'frozen_embeddings' pre-chunked, it might already be shape
            # (batch_size, num_chunks, embed_dim). If so, skip the chunk function.
            # Otherwise, we do chunking from input_ids. 
            if frozen_embeddings is not None:
                # We expect shape (B, num_chunks, embed_dim). 
                # If it's only (B, embed_dim), you'd need to handle that differently or raise an error.
                if len(embeddings.shape) == 2:
                    raise ValueError(
                        "For overlap_chunking with frozen embeddings, pass them in chunked form: (B, num_chunks, embed_dim)."
                    )
                chunked_embeddings = embeddings  # shape => (B, num_chunks, embed_dim)
            else:
                # Use your chunking helper to convert (B, seq_len) => (B, num_chunks, embed_dim)
                chunked_embeddings = self._chunk_and_embed(
                    embedder_input_ids, embedder_attention_mask
                )
                # 'embeddings' from call_embedding_model(...) is not used directly here,
                # because chunking means we do multiple calls inside _chunk_and_embed.

            B, N, E = chunked_embeddings.shape  # batch, num_chunks, embed_dim
            # Flatten chunk dimension for an MLP pass
            flat_chunked = chunked_embeddings.reshape(B * N, E)

            # Pass through bridging MLP, which outputs hidden_dim
            transformed_2d = self.embedding_transform(flat_chunked)
            # => shape: (B*N, hidden_dim)

            # Reshape to (batch_size, num_chunks, hidden_dim)
            final_embeddings = transformed_2d.reshape(B, N, -1)

            # Build an attention mask: 1 means "attend to this chunk"
            attention_mask = torch.ones(
                (B, N), dtype=torch.long, device=final_embeddings.device
            )

        elif self.embedding_transform_strategy == "repeat":
            # (B) Old logic: single embedding => repeat
            # embeddings => shape (batch_size, embed_dim)
            # self.embedding_transform => yields (batch_size, hidden_dim * num_repeat_tokens)
            repeated_embeddings = self.embedding_transform(embeddings)
            # Reshape to (batch_size, num_repeat_tokens, hidden_dim)
            final_embeddings = repeated_embeddings.reshape(
                (*repeated_embeddings.shape[:-1], self.num_repeat_tokens, -1)
            )
            # Build attention mask
            attention_mask = torch.ones(
                (final_embeddings.shape[0], final_embeddings.shape[1]),
                dtype=torch.long,
                device=final_embeddings.device,
            )

        else:
            # Unknown strategy
            raise ValueError(
                f"Unknown embedding_transform_strategy: {self.embedding_transform_strategy}"
            )

        return final_embeddings, attention_mask

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Unused: input_ids, attention_mask
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
