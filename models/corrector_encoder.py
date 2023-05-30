import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers


class CorrectorEncoderModel(torch.nn.Module):
    """Embeds text and concats with a provided embedding.

    TODO improve comment here.
    """

    encoder_decoder: transformers.PreTrainedModel

    def __init__(
        self,
        encoder_decoder: transformers.PreTrainedModel,
        embedder_dim: int = 768,
        num_repeat_tokens: int = 16,
        bottleneck_dim: int = 768,
        ignore_hypothesis_embedding: bool = False,
    ):
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.embedder_dim = embedder_dim
        self.num_repeat_tokens = num_repeat_tokens
        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedding_transform_1 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        )
        self.embedding_transform_2 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        )
        self.embedding_transform_3 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        )
        self.ignore_hypothesis_embedding = ignore_hypothesis_embedding

    def get_encoder_embedding(
        self,
        embedding: torch.Tensor,
        hypothesis_embedding: torch.Tensor,
        hypothesis_input_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, D = embedding.shape
        assert embedding.shape == (batch_size, 768)
        assert hypothesis_embedding.shape == (batch_size, 768)

        if self.ignore_hypothesis_embedding:
            hypothesis_embedding = embedding

        diff_embedding = embedding - hypothesis_embedding

        embedding = self.embedding_transform_1(embedding)
        embedding = embedding.reshape((batch_size, self.num_repeat_tokens, -1))
        #
        diff_embedding = self.embedding_transform_2(diff_embedding)
        diff_embedding = diff_embedding.reshape((batch_size, self.num_repeat_tokens, D))
        #
        hypothesis_embedding = self.embedding_transform_3(hypothesis_embedding)
        hypothesis_embedding = hypothesis_embedding.reshape(
            (batch_size, self.num_repeat_tokens, -1)
        )
        inputs_embeds = self.encoder_decoder.encoder.embed_tokens(hypothesis_input_ids)
        #
        ones = torch.ones(
            (batch_size, 1), dtype=torch.long, device=hypothesis_input_ids.device
        )
        # TODO: pad_token_id or eos_token_id? Or does it not matter?
        sep_token = ones * self.encoder_decoder.config.eos_token_id
        sep_token = self.encoder_decoder.encoder.embed_tokens(sep_token)
        # inputs_embeds = torch.cat((sep_token, embedding, sep_token, hypothesis_embedding, inputs_embeds), dim=1)
        inputs_embeds = torch.cat(
            (
                sep_token,
                embedding,
                sep_token,
                hypothesis_embedding,
                sep_token,
                diff_embedding,
                sep_token,
                inputs_embeds,
            ),
            dim=1,
        )
        attention_mask = torch.cat(
            (ones.repeat(1, 4 + 3 * self.num_repeat_tokens), hypothesis_attention_mask),
            dim=1,
        )
        return (inputs_embeds, attention_mask)

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "max_length" not in generation_kwargs:
            generation_kwargs = copy.copy(
                generation_kwargs
            )  # make a copy so we can edit
            generation_kwargs["max_length"] = inputs.get(
                "input_ids", inputs["embedder_input_ids"]
            ).shape[1]

        # print("CE.generate:", generation_kwargs)

        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=inputs["frozen_embeddings"],
            hypothesis_input_ids=inputs["hypothesis_input_ids"],
            hypothesis_attention_mask=inputs["hypothesis_attention_mask"],
            hypothesis_embedding=inputs["hypothesis_embedding"],
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
        embedding: torch.Tensor,
        hypothesis_embedding,
        hypothesis_input_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        inputs_embeds, attention_mask = self.get_encoder_embedding(
            embedding=embedding,
            hypothesis_embedding=hypothesis_embedding,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )
        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )