# models/inversion.py

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
    disable_dropout,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    load_tokenizer,
    mean_pool,
)
from vec2text.utils import embed_api

logger = logging.getLogger(__name__)


# (Optional) We'll keep a refinement block for multi-step decode at inference
class RefinementBlock(nn.Module):
    """
    A small sub-network that refines a previously generated text. 
    We'll do multi-step chunk-based or a single pass for the refine stage.
    """

    def __init__(self, config: InversionConfig):
        super().__init__()
        self.config = config

        # A small seq2seq sub-model for refinement
        self.encoder_decoder = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path
        )

        embedder_dim = 1536 if config.embedder_model_api else 768
        hidden_size = self.encoder_decoder.config.hidden_size

        self.embedding_transform = nn.Sequential(
            nn.Linear(embedder_dim, embedder_dim),
            nn.GELU(),
            nn.Linear(embedder_dim, hidden_size),
        )
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward_generate(
        self,
        frozen_embeddings: torch.Tensor,
        partial_embeddings: torch.Tensor,
        partial_ids: torch.Tensor,
        partial_mask: torch.Tensor,
        generation_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        B = frozen_embeddings.size(0)
        device = frozen_embeddings.device

        # transform + combine
        fe = self.embedding_transform(frozen_embeddings)
        pe = self.embedding_transform(partial_embeddings)
        combined = self.layernorm(fe + pe).unsqueeze(1)  # [B,1, hidden_size]

        # embed partial tokens
        input_embs = self.encoder_decoder.get_encoder().embed_tokens(partial_ids) 
        new_inputs_embs = torch.cat([combined, input_embs], dim=1).contiguous()

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
    A model that:
      - (Training) does chunk-level teacher-forced decode with partial re-embedding
        and an embedding alignment aux-loss.
      - (Inference) can do single pass or multi-step refinement (optional).

    We unify strong lexical cross-entropy + chunk re-embedding + optional embedding-sim alignment.
    """

    config_class = InversionConfig

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        # 1) Load base seq2seq
        encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )

        # 2) Load embedder
        embedder, embedder_tokenizer = load_embedder_and_tokenizer(
            name=config.embedder_model_name,
            torch_dtype=config.embedder_torch_dtype,
        )
        tokenizer = load_tokenizer(
            config.model_name_or_path,
            max_length=config.max_seq_length,
        )

        self.encoder_decoder = encoder_decoder
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.embedder_tokenizer = embedder_tokenizer

        # Decide dims
        if config.embedder_model_api:
            self.embedder_dim = 1536
            bottleneck_dim = 1536
        elif isinstance(embedder, SentenceTransformer):
            self.embedder_dim = embedder.get_sentence_embedding_dimension()
            bottleneck_dim = self.embedder_dim
        else:
            self.embedder_dim = embedder.config.hidden_size
            bottleneck_dim = self.embedder_dim

        hidden_size = self.encoder_decoder.config.hidden_size
        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_size),
        )

        # disable dropouts if asked
        if config.encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if config.decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)

        # For good measure
        self.embedding_layernorm = nn.LayerNorm(hidden_size)

        # store
        self.noise_level = getattr(config, "embedder_gaussian_noise_level", 0.0)
        self.embedder_no_grad = config.embedder_no_grad
        if self.embedder_no_grad:
            for p in self.embedder.parameters():
                p.requires_grad = False
            self.embedder.eval()

        # (Optional) a refinement sub-block for multi-step decode at inference
        self.refinement_block = RefinementBlock(config)

        # (Hyperparams) For chunk training
        self.train_num_chunks = getattr(config, "train_num_chunks", 4)  # how many chunks to break the label text into
        self.embedding_sim_scale = 0.1  # weighting factor for embedding alignment loss

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> transformers.modeling_outputs.Seq2SeqLMOutput:
        """
        We do a chunk-level teacher forcing:
         - split labels into chunks
         - decode each chunk in teacher-forced manner
         - re-embed partial decode -> measure alignment to target emb
         - sum up cross-entropy from each chunk + embedding alignment
         - return final

        If no labels are provided, we do a normal pass (like your old single-shot).
        """

        device = embedder_input_ids.device

        # 1) embed the input
        with torch.no_grad():
            emb = self._call_embedding_model(
                embedder_input_ids, embedder_attention_mask
            )
        # project
        base_hidden = self.embedding_transform(emb)
        base_hidden = self.embedding_layernorm(base_hidden)  # shape [B, hidden_size]

        # If no labels, do the old single pass
        if labels is None:
            # standard forward => compute hidden => forward in seq2seq
            # We'll just treat it as we do in generate, returning dummy
            return self._forward_no_labels(base_hidden, **kwargs)

        # If we have labels => chunk-level teacher forcing approach
        # 2) We'll chunk the label sequence. 
        #    For simplicity, we treat each chunk as a small "sub-sequence."

        # We'll do a naive approach: we just chunk the label tokens in dimension=1
        # e.g. labels shape: [B, seqLen]
        # We want train_num_chunks pieces. We must ensure it doesn't exceed seqLen
        B, full_seq_len = labels.shape
        chunk_size = max(1, full_seq_len // self.train_num_chunks)

        total_ce_loss = 0.0
        total_emb_loss = 0.0
        # We'll keep the partial decode
        partial_ids = None

        for c_idx in range(self.train_num_chunks):
            start_idx = c_idx * chunk_size
            end_idx = min(full_seq_len, start_idx + chunk_size)
            if start_idx >= end_idx:
                break

            # Sub-chunk teacher forcing 
            # We'll treat chunk_labels as the "next tokens" we want to produce
            chunk_labels = labels[:, start_idx:end_idx]  # shape [B, chunk_len]

            # Build a "decoder_input_ids" 
            # Typically, for teacher forcing, we shift chunk_labels right by 1. 
            # For simplicity, let's do it naive: the chunk_labels are the entire forced sequence, 
            # letting HF handle shift inside the model. 
            # We also pass an "encoder input" = base_hidden, plus partial decode?

            # We'll embed partial decode so far -> build input_embs
            if partial_ids is not None:
                # re-embed the partial decode -> partial_emb
                partial_emb = self._embed_partial_ids(partial_ids)
                # combine partial_emb with base_hidden
                combined = (base_hidden + partial_emb) * 0.5
            else:
                combined = base_hidden

            # shape => [B,1, hidden_size]
            combined = combined.contiguous()
            combined = combined.unsqueeze(1)

            # We'll pass combined as an "inputs_embeds" for the encoder
            # For T5 or BART, we might do a tiny hack: 
            #   "encoder_inputs_embeds = combined" & 
            #   "decoder_input_ids = chunk_labels"
            # This is a simplified approach. We'll do a small workaround:

            # "Fake" an attention mask = ones
            enc_attn_mask = torch.ones((B,1), dtype=torch.long, device=device)
            # We'll feed chunk_labels as labels => we get cross-entropy
            outputs = self.encoder_decoder(
                inputs_embeds=combined,       # shape [B,1, hidden_size]
                attention_mask=enc_attn_mask,
                labels=chunk_labels,
            )
            # Cross-entropy from this chunk
            ce_loss = outputs.loss

            # We'll do partial decode = argmax next tokens
            with torch.no_grad():
                # We can do "teacher forcing => partial_ids = chunk_labels" 
                # for a simpler approach. So partial_ids = chunk_labels 
                # actually means we 'pretend' the model produced it. 
                # Or we can do "argmax from outputs.logits." 
                # We'll do teacher forcing approach:
                partial_ids_this = chunk_labels  
            if partial_ids is None:
                partial_ids = partial_ids_this
            else:
                # concat 
                partial_ids = torch.cat([partial_ids, partial_ids_this], dim=1)

            # 3) compute an embedding alignment with the final "partial decode" so far
            with torch.no_grad():
                partial_emb = self._embed_partial_ids(partial_ids)  # shape [B, embedder_dim]
            # measure distance to original "emb"
            # e.g. L2 or 1 - cos sim
            cos = F.cosine_similarity(partial_emb, emb, dim=-1)
            # we'd like cos ~ 1
            emb_loss = (1.0 - cos).mean()

            total_ce_loss += ce_loss
            total_emb_loss += emb_loss

        # Average them across chunks
        ce_final = total_ce_loss / self.train_num_chunks
        emb_final = total_emb_loss / self.train_num_chunks
        total_loss = ce_final + self.embedding_sim_scale * emb_final

        # We'll produce a standard HF Seq2SeqLMOutput
        return transformers.modeling_outputs.Seq2SeqLMOutput(
            loss=total_loss,
            logits=None,  # We don't produce a final logits
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    def _forward_no_labels(self, base_hidden: torch.Tensor, **kwargs):
        """
        If user calls forward without labels, 
        we just do a 1-token pass or no pass. 
        We won't produce a meaningful loss. 
        """
        # do a minimal approach => return no loss
        return transformers.modeling_outputs.Seq2SeqLMOutput(
            loss=None, 
            logits=None
        )

    def _call_embedding_model(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # same as call_embedding_model, but inlined
        if self.embedder_no_grad:
            self.embedder.eval()
        model_output = None

        # If user used embedder_fake_with_zeros, skip
        # or if config.embedder_model_api => call remote
        # for brevity, we do local st approach
        if isinstance(self.embedder, SentenceTransformer):
            with torch.no_grad():
                out = self.embedder(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                )
            emb = out["sentence_embedding"]
        else:
            # typical huggingface model
            out = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
            # we do a pool or last hidden
            emb = mean_pool(out.last_hidden_state, attention_mask)

        if self.noise_level > 0.0 and self.training:
            emb = emb + self.noise_level*torch.randn_like(emb)
        return emb

    def _embed_partial_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Re-embed partial text from token_ids => get an embedding of shape [B, embedder_dim].
        If user wants chunk-lens, we flatten or do something. 
        We'll do a naive approach: decode to strings & re-embed.
        """
        device = token_ids.device
        txts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        emb_inp = self.embedder_tokenizer(
            txts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            out = self.embedder(emb_inp)
        partial_emb = out["sentence_embedding"] if isinstance(self.embedder, SentenceTransformer) else mean_pool(
            out.last_hidden_state, emb_inp["attention_mask"]
        )
        return partial_emb

    # ---------------------------------------------------------------------
    # 2) The final .generate() with multi-step refinement
    # ---------------------------------------------------------------------
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any],
        refine_steps: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Single pass or multi-step chunk refine approach
        """
        device = next(self.parameters()).device

        # 1) single pass => do normal base decode
        base_ids = self._single_pass_generate(inputs, generation_kwargs)

        if refine_steps <= 0:
            return base_ids

        # 2) iterative refine using self.refinement_block
        current_ids = base_ids
        frozen_embeddings = inputs["frozen_embeddings"].to(device)

        for step_idx in range(refine_steps):
            partial_emb = self._embed_partial_ids(current_ids)
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

    def _single_pass_generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        The same old single-pass approach: 
         embed => project => run base encoder_decoder.generate
        """
        device = next(self.parameters()).device

        # get embedding
        if "frozen_embeddings" in inputs:
            base_emb = inputs["frozen_embeddings"].to(device)
        else:
            # embed from scratch
            base_emb = self._call_embedding_model(
                inputs["embedder_input_ids"], 
                inputs["embedder_attention_mask"]
            )
        # project
        hidden = self.embedding_transform(base_emb)
        hidden = self.embedding_layernorm(hidden)

        # shape => [B,1,hidden_size]
        B, hidden_size = hidden.shape
        hidden = hidden.unsqueeze(1)

        # we do a trivial attention mask => ones
        attn_mask = torch.ones((B,1), dtype=torch.long, device=device)

        return self.encoder_decoder.generate(
            inputs_embeds=hidden,
            attention_mask=attn_mask,
            **generation_kwargs
        )
