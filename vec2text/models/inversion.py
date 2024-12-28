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

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        )
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

        self.embedding_transform_strategy = "repeat"  # "none" # "repeat"
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = vars(config).get("embedder_gaussian_noise_level")

    def _freeze_encoder(self):
        freeze_params(self.encoder_decoder.encoder)

    def _freeze_decoder(self):
        # github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L1229-L1231
        freeze_params(self.encoder_decoder.decoder)
        freeze_params(self.encoder_decoder.lm_head)

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
        # print("** embed_and_project")
        assert not ((embedder_input_ids is None) and (frozen_embeddings is None))
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2  # batch by d
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
        if self.embedding_transform_strategy == "repeat":
            if embeddings.dtype != self.dtype:
                embeddings = embeddings.to(self.dtype)
            repeated_embeddings = self.embedding_transform(embeddings)
            # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
            embeddings = repeated_embeddings.reshape(
                (*repeated_embeddings.shape[:-1], self.num_repeat_tokens, -1)
            )
        elif self.embedding_transform_strategy == "nearest_neighbors":
            # TODO
            raise NotImplementedError()
        else:
            raise ValueError(
                f"unknown embedding transformation strategy {self.embedding_transform_strategy}"
            )
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )
        return embeddings, attention_mask

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
        guided: bool = False,
        checkpoint_interval: int = 10,
        beam_size: int = 5,
        multipass: int = 1,
        alpha: float = 1.0,
        beta: float = 1.0,
        length_penalty: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        A unified .generate() that can do:
            - Normal single-pass (guided=False, multipass=1)
            - Single-pass partial-chunk/beam-checkpoint logic (guided=True, multipass=1)
            - Multi-pass iterative refinement with partial-chunk beam search (guided=True, multipass>1)

        Args:
            inputs: e.g. {"frozen_embeddings": ..., or "embedder_input_ids": ...}
            generation_kwargs: typical HF generation params 
                               (max_length, do_sample, etc.)
            guided: if True, use partial-chunk beam-checkpoint approach
            checkpoint_interval: how often (in tokens) to do embedding-based checkpoint
            beam_size: how large the beam is
            multipass: how many times we do a full pass of generation, re-embed, decode again
            alpha, beta: weighting for LM log-prob vs. embedding similarity
            length_penalty: penalize longer sequences in scoring
        """
        # 1) If guided=False and multipass=1 => do your existing single pass
        if not guided and multipass == 1:
            return self._single_pass_generate(inputs=inputs, generation_kwargs=generation_kwargs)

        # 2) If multipass=1 but guided=True => do partial-chunk beam approach once
        if multipass == 1:
            return self._guided_beam_search_single_pass(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                checkpoint_interval=checkpoint_interval,
                beam_size=beam_size,
                alpha=alpha,
                beta=beta,
                length_penalty=length_penalty,
            )

        # 3) Otherwise, multipass>1 => we do multiple guided passes in a loop
        #    We'll do the partial-chunk beam approach in each pass, then re-embed.
        return self._multipass_guided_decode(
            inputs=inputs,
            generation_kwargs=generation_kwargs,
            num_passes=multipass,
            checkpoint_interval=checkpoint_interval,
            beam_size=beam_size,
            alpha=alpha,
            beta=beta,
            length_penalty=length_penalty,
        )

    def _single_pass_generate(
        self, 
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Your old single-pass approach: embed_and_project -> encoder_decoder.generate(...)
        """
        # embed & project
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

    def _guided_beam_search_single_pass(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
        checkpoint_interval: int,
        beam_size: int,
        alpha: float,
        beta: float,
        length_penalty: float,
    ) -> torch.Tensor:
        """
        A single decoding pass with partial-chunk/beam-checkpoint logic:
          1) Expand partials, 
          2) every `checkpoint_interval` tokens, embed partials, combine similarity with LM log-prob, prune beam
          3) continue until max_new_tokens or EOS
        We'll produce the best final output (token IDs).
        """
        device = self.device
        # We need to embed_and_project to get the "inputs_embeds" for the T5 encoder
        # but we're going to manually do the decoding side ourselves, in a custom loop
        # if we're fully overriding the beam. Alternatively, we can do partial chunk logic by 
        # hooking into each token step. This code is a high-level sample.

        # Let's do a simpler approach: we'll replicate the method from 
        # "guided beam search" snippet. 
        # For brevity, see that snippet for details. We'll just show a short version:

        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )

        # Assume batch_size=1 for simplicity
        start_token_id = self.tokenizer.pad_token_id
        eos_token_id   = self.tokenizer.eos_token_id
        beam = [{
            "tokens": [start_token_id],
            "lm_log_prob": 0.0,
        }]
        finished = []

        max_new_tokens = generation_kwargs.get("max_new_tokens", 50)

        for step in range(max_new_tokens):
            new_candidates = []
            for hyp in beam:
                if hyp["tokens"][-1] == eos_token_id:
                    new_candidates.append(hyp)
                    continue

                # build decoder_input_ids
                decoder_input_ids = torch.tensor([hyp["tokens"]], device=device)
                with torch.no_grad():
                    out = self.encoder_decoder(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                    )
                next_logits = out.logits[0, -1, :]
                next_probs = torch.softmax(next_logits, dim=-1)
                topk = torch.topk(next_probs, beam_size)
                for tk_id, tk_prob in zip(topk.indices.tolist(), topk.values.tolist()):
                    new_seq = hyp["tokens"] + [tk_id]
                    new_lp  = hyp["lm_log_prob"] + math.log(tk_prob + 1e-9)
                    new_candidates.append({
                        "tokens": new_seq,
                        "lm_log_prob": new_lp,
                    })

            # prune quickly by LM log-prob
            new_candidates.sort(key=lambda c: c["lm_log_prob"], reverse=True)
            new_candidates = new_candidates[: beam_size * 2]

            # checkpoint?
            is_checkpoint = ((step + 1) % checkpoint_interval == 0) or (step == max_new_tokens - 1)
            if is_checkpoint:
                # embed partial sequences, combine with lm_log_prob
                scored = []
                for cand in new_candidates:
                    if cand["tokens"][-1] == eos_token_id:
                        # no partial embed
                        cand["combined_score"] = cand["lm_log_prob"]  
                        scored.append(cand)
                    else:
                        # decode to text
                        txt = self.tokenizer.decode(cand["tokens"], skip_special_tokens=True)
                        emb_inp = self.embedder_tokenizer(txt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            partial_emb = self.call_embedding_model(
                                input_ids=emb_inp["input_ids"],
                                attention_mask=emb_inp["attention_mask"],
                            )
                        # compare partial_emb to target, e.g. if "frozen_embeddings" was your target
                        # if you have a known "target_embedding" from the inputs, e.g. inputs["target_embedding"]
                        # do cos_sim = ...
                        target_emb = inputs.get("target_embedding", None)
                        if target_emb is not None:
                            cos_sim = torch.nn.functional.cosine_similarity(
                                partial_emb, target_emb.unsqueeze(0), dim=-1
                            )
                            cand["embed_score"] = cos_sim.item()
                        else:
                            cand["embed_score"] = 0.0
                        # combine
                        seq_len = len(cand["tokens"])
                        lp = (5 + seq_len) / 6.0
                        cand["combined_score"] = (
                            alpha * (cand["lm_log_prob"] / (lp ** length_penalty)) +
                            beta  * cand["embed_score"]
                        )
                        scored.append(cand)

                scored.sort(key=lambda c: c["combined_score"], reverse=True)
                scored = scored[: beam_size]
                # separate out ended
                beam = []
                for s in scored:
                    if s["tokens"][-1] == eos_token_id:
                        finished.append(s)
                    else:
                        beam.append(s)
            else:
                # no checkpoint => keep top beam_size by LM log prob
                new_candidates = new_candidates[: beam_size]
                next_beam = []
                for c in new_candidates:
                    if c["tokens"][-1] == eos_token_id:
                        finished.append(c)
                    else:
                        next_beam.append(c)
                beam = next_beam

            if not beam:
                break

        finished.extend(beam)
        # pick best final by combined_score or lm_log_prob if it never got a checkpoint
        for f in finished:
            if "combined_score" not in f:
                seq_len = len(f["tokens"])
                lp = (5 + seq_len) / 6.0
                f["combined_score"] = f["lm_log_prob"] / (lp ** length_penalty)
        finished.sort(key=lambda c: c["combined_score"], reverse=True)
        best = finished[0]["tokens"]
        return torch.tensor([best], device=device, dtype=torch.long)

    def _multipass_guided_decode(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
        num_passes: int,
        checkpoint_interval: int,
        beam_size: int,
        alpha: float,
        beta: float,
        length_penalty: float,
    ) -> torch.Tensor:
        """
        Repeatedly do guided beam search, re-embed the best final output each time,
        feed that embedding back in as the new input for next pass.
        """
        device = self.device

        # 1) Pass #1
        best_output_ids = self._guided_beam_search_single_pass(
            inputs=inputs,
            generation_kwargs=generation_kwargs,
            checkpoint_interval=checkpoint_interval,
            beam_size=beam_size,
            alpha=alpha,
            beta=beta,
            length_penalty=length_penalty,
        )

        # 2) If we only had 1 pass, we're done
        if num_passes <= 1:
            return best_output_ids

        # 3) For additional passes, re-embed best output each time
        for pass_idx in range(1, num_passes):
            # decode best output => text
            txts = self.tokenizer.batch_decode(best_output_ids, skip_special_tokens=True)
            new_embs = []
            for t in txts:
                emb_inp = self.embedder_tokenizer(t, return_tensors="pt").to(device)
                with torch.no_grad():
                    e = self.call_embedding_model(
                        input_ids=emb_inp["input_ids"],
                        attention_mask=emb_inp["attention_mask"]
                    )
                new_embs.append(e)
            new_embs = torch.cat(new_embs, dim=0) # shape (batch, embed_dim)

            # feed that in as frozen_embeddings to next pass
            new_inputs = copy.copy(inputs)
            new_inputs["frozen_embeddings"] = new_embs

            # re-generate with partial-chunk beam search
            best_output_ids = self._guided_beam_search_single_pass(
                inputs=new_inputs,
                generation_kwargs=generation_kwargs,
                checkpoint_interval=checkpoint_interval,
                beam_size=beam_size,
                alpha=alpha,
                beta=beta,
                length_penalty=length_penalty,
            )

        return best_output_ids

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
