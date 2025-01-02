import copy
import logging
from typing import Dict, Optional, Tuple

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
        print("[embed_and_project] Entered method")
        assert not ((embedder_input_ids is None) and (frozen_embeddings is None))

        # Step 1: Decide whether we're using precomputed/frozen embeddings vs. calling the embedder.
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2, (
                f"Expected frozen_embeddings to be rank 2 (batch, dim), got shape {embeddings.shape}"
            )
            print(f"[embed_and_project] Using frozen_embeddings; shape={embeddings.shape}")
        elif self.embedder_no_grad:
            print("[embed_and_project] embedder_no_grad=True, calling embedding model in no_grad mode")
            with torch.no_grad():
                embeddings = self.call_embedding_model(
                    input_ids=embedder_input_ids,
                    attention_mask=embedder_attention_mask,
                )
            print(f"[embed_and_project] embeddings shape after embedder_no_grad call = {embeddings.shape}")
        else:
            print("[embed_and_project] embedder_no_grad=False, calling embedding model with grad")
            embeddings = self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )
            print(f"[embed_and_project] embeddings shape after embedder call = {embeddings.shape}")

        # Step 2: If we're using the "repeat" strategy, do linear transform and reshape into a sequence.
        if self.embedding_transform_strategy == "repeat":
            if embeddings.dtype != self.dtype:
                embeddings = embeddings.to(self.dtype)
                print(f"[embed_and_project] Converted embeddings dtype from {embeddings.dtype} to {self.dtype}")
            repeated_embeddings = self.embedding_transform(embeddings)
            print(f"[embed_and_project] repeated_embeddings shape after linear transform = {repeated_embeddings.shape}")

            # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
            print(f"[embed_and_project] About to reshape with num_repeat_tokens={self.num_repeat_tokens}")
            embeddings = repeated_embeddings.reshape(
                (*repeated_embeddings.shape[:-1], self.num_repeat_tokens, -1)
            )
            print(f"[embed_and_project] embeddings shape after reshape = {embeddings.shape}")
        elif self.embedding_transform_strategy == "nearest_neighbors":
            # TODO
            raise NotImplementedError("[embed_and_project] nearest_neighbors strategy not implemented")
        else:
            raise ValueError(
                f"[embed_and_project] Unknown embedding transformation strategy: {self.embedding_transform_strategy}"
            )

        # Step 3: Build an attention mask (all ones) that matches the first two dims of `embeddings`.
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )
        print(f"[embed_and_project] Created attention_mask with shape={attention_mask.shape}")

        print("[embed_and_project] Exiting method\n")
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
        # Pull out any custom arguments from generation_kwargs if present
        if "guided" in generation_kwargs:
            guided = generation_kwargs.pop("guided")
        if "checkpoint_interval" in generation_kwargs:
            checkpoint_interval = generation_kwargs.pop("checkpoint_interval")
        if "beam_size" in generation_kwargs:
            beam_size = generation_kwargs.pop("beam_size")
        if "multipass" in generation_kwargs:
            multipass = generation_kwargs.pop("multipass")
        if "alpha" in generation_kwargs:
            alpha = generation_kwargs.pop("alpha")
        if "beta" in generation_kwargs:
            beta = generation_kwargs.pop("beta")
        if "length_penalty" in generation_kwargs:
            length_penalty = generation_kwargs.pop("length_penalty")

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
        Old single-pass approach: embed_and_project -> encoder_decoder.generate(...)
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
        A single decoding pass with partial-chunk/beam-checkpoint logic,
        now generalized to arbitrary batch_size B.

        We'll maintain B groups of beam_size hypotheses each, for a total
        of (B * beam_size) partial sequences in parallel.
        """

        import math
        import torch
        import torch.nn.functional as F

        device = self.device

        # 1) Encode once to get inputs_embeds for T5 of shape: [B, enc_seq_len, hidden_dim].
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
        batch_size = inputs_embeds.size(0)

        # 2) Repeat (tile) the encoder inputs across beam_size => [B*beam_size, enc_seq_len, hidden_dim].
        #    Also repeat the attention_mask => [B*beam_size, enc_seq_len].
        inputs_embeds = inputs_embeds.unsqueeze(1).repeat(1, beam_size, 1, 1)  # [B, beam_size, enc_seq_len, hidden_dim]
        inputs_embeds = inputs_embeds.view(batch_size * beam_size, *inputs_embeds.shape[2:])

        attention_mask = attention_mask.unsqueeze(1).repeat(1, beam_size, 1)  # [B, beam_size, enc_seq_len]
        attention_mask = attention_mask.view(batch_size * beam_size, -1)

        # 3) Prepare the initial beam states:
        #    We'll have a total of B*beam_size "rows" in the batch dimension,
        #    each one storing partial tokens, log-prob, etc.
        #    We'll store them in a list "beam_hyps" of length (B * beam_size).
        #    Each entry: {
        #        "batch_idx": i,  # which *original* example in [0..B-1]
        #        "tokens": [...],
        #        "lm_log_prob": 0.0,
        #    }
        start_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        beam_hyps = []
        for batch_i in range(batch_size):
            for beam_j in range(beam_size):
                beam_hyps.append({
                    "batch_idx": batch_i,     # which original example?
                    "tokens": [start_token_id],
                    "lm_log_prob": 0.0,
                    # "combined_score" is set only at checkpoint
                })

        finished = [[] for _ in range(batch_size)]  # store finished hyps per example

        # We'll read HF generation kwargs for typical parameters (e.g. max_new_tokens)
        max_new_tokens = generation_kwargs.get("max_new_tokens", 50)

        # If we have "target_embedding" => shape [B, emb_dim]
        # We'll use it to compare partial text embeddings during checkpoints
        batch_target_emb = inputs.get("target_embedding", None)  # Optional[Tensor], shape [B, emb_dim]

        # 4) Main decoding loop
        for step in range(max_new_tokens):
            # (a) Build decoder_input_ids => shape [B*beam_size, dec_seq_len_so_far].
            #     We gather the partial 'tokens' from each of the B*beam_size beam items.
            max_len = max(len(hyp["tokens"]) for hyp in beam_hyps)
            decoder_input_ids_list = []
            for hyp in beam_hyps:
                needed = max_len - len(hyp["tokens"])
                padded_tokens = hyp["tokens"] + [start_token_id] * needed
                decoder_input_ids_list.append(padded_tokens)
            decoder_input_ids = torch.tensor(decoder_input_ids_list, device=device)

            print("[guided_beam_search] =====================")
            print(f"  batch_size * beam_size = {batch_size * beam_size}")  # if you have those variables
            print(f"  inputs_embeds.shape = {inputs_embeds.shape}")
            print(f"  attention_mask.shape = {attention_mask.shape}")
            print(f"  decoder_input_ids.shape = {decoder_input_ids.shape}")
            print("[guided_beam_search] =====================")

            # (b) Forward pass (one big batch of size B*beam_size)
            with torch.no_grad():
                out = self.encoder_decoder(
                    inputs_embeds=inputs_embeds,        # [B*beam_size, enc_seq_len, hidden_dim]
                    attention_mask=attention_mask,       # [B*beam_size, enc_seq_len]
                    decoder_input_ids=decoder_input_ids, # [B*beam_size, dec_seq_len_so_far]
                )
            # out.logits => shape [B*beam_size, dec_seq_len_so_far, vocab_size]
            next_logits = out.logits[:, -1, :]  # => [B*beam_size, vocab_size]

            # (c) We'll expand each example's beams separately, so we don't mix beam expansions across examples.
            #     We'll group the B*beam_size rows by example => "grouped" by batch_idx in [0..B-1].
            #     Then for each group => we do topk expansions. We'll accumulate expansions into next_beam_hyps_list.
            next_beam_hyps_list = [[] for _ in range(batch_size)]

            for row_idx, hyp in enumerate(beam_hyps):
                batch_i = hyp["batch_idx"]
                # If already ended => carry forward
                if hyp["tokens"][-1] == eos_token_id:
                    next_beam_hyps_list[batch_i].append(hyp)
                    continue

                # Expand by top-k
                row_probs = torch.softmax(next_logits[row_idx], dim=-1)
                topk = torch.topk(row_probs, beam_size)
                for tk_id, tk_prob in zip(topk.indices.tolist(), topk.values.tolist()):
                    new_seq = hyp["tokens"] + [tk_id]
                    new_lp = hyp["lm_log_prob"] + math.log(tk_prob + 1e-9)
                    next_beam_hyps_list[batch_i].append({
                        "batch_idx": batch_i,
                        "tokens": new_seq,
                        "lm_log_prob": new_lp,
                    })

            # (d) Now for each example's group, we do "quick beam pruning" by LM log-prob => keep top (beam_size * 2)
            #     so we don't blow up in size. We'll store them back in the same next_beam_hyps_list[batch_i].
            for batch_i in range(batch_size):
                group_cands = next_beam_hyps_list[batch_i]
                group_cands.sort(key=lambda c: c["lm_log_prob"], reverse=True)
                group_cands = group_cands[: beam_size * 2]
                next_beam_hyps_list[batch_i] = group_cands

            # (e) If it's a checkpoint step, re-embed partial text for each group candidate.
            #     Then combine LM log-prob w/ embedding similarity => "combined_score".
            is_checkpoint = ((step + 1) % checkpoint_interval == 0) or (step == max_new_tokens - 1)

            if is_checkpoint:
                # We'll produce the new pruned list for each example after embed-based scoring.
                new_beam_hyps_list = [[] for _ in range(batch_size)]
                for batch_i in range(batch_size):
                    scored = []
                    for cand in next_beam_hyps_list[batch_i]:
                        # If ended => no partial embed
                        if cand["tokens"][-1] == eos_token_id:
                            cand["combined_score"] = cand["lm_log_prob"]
                            scored.append(cand)
                        else:
                            # decode partial text
                            txt = self.tokenizer.decode(cand["tokens"], skip_special_tokens=True)
                            # re-embed partial text => shape [1, embed_dim]
                            emb_inp = self.embedder_tokenizer(txt, return_tensors="pt").to(device)
                            with torch.no_grad():
                                partial_emb = self.call_embedding_model(
                                    input_ids=emb_inp["input_ids"],
                                    attention_mask=emb_inp["attention_mask"],
                                )
                            # If we have a batch target embedding => get the row = target_emb[batch_i]
                            embed_score = 0.0
                            if batch_target_emb is not None:
                                # shape => [embed_dim]
                                example_target = batch_target_emb[batch_i]
                                cos_sim = F.cosine_similarity(partial_emb, example_target.unsqueeze(0), dim=-1)
                                embed_score = cos_sim.item()

                            seq_len = len(cand["tokens"])
                            lp = (5 + seq_len) / 6.0
                            cand["embed_score"] = embed_score
                            cand["combined_score"] = (
                                alpha * (cand["lm_log_prob"] / (lp ** length_penalty))
                                + beta * embed_score
                            )
                            scored.append(cand)

                    # re-sort by combined_score => keep top beam_size
                    scored.sort(key=lambda c: c["combined_score"], reverse=True)
                    scored = scored[: beam_size]

                    # separate out ended
                    still_going = []
                    for s in scored:
                        if s["tokens"][-1] == eos_token_id:
                            finished[batch_i].append(s)
                        else:
                            still_going.append(s)
                    new_beam_hyps_list[batch_i] = still_going

                # Overwrite next_beam_hyps_list with new_beam_hyps_list
                next_beam_hyps_list = new_beam_hyps_list
            else:
                # Not a checkpoint => keep top beam_size by LM log-prob, then remove ended from beam
                new_beam_hyps_list = [[] for _ in range(batch_size)]
                for batch_i in range(batch_size):
                    group_cands = next_beam_hyps_list[batch_i][: beam_size]
                    still_going = []
                    for c in group_cands:
                        if c["tokens"][-1] == eos_token_id:
                            finished[batch_i].append(c)
                        else:
                            still_going.append(c)
                    new_beam_hyps_list[batch_i] = still_going
                next_beam_hyps_list = new_beam_hyps_list

            # If all groups are empty => done
            all_empty = True
            for batch_i in range(batch_size):
                if len(next_beam_hyps_list[batch_i]) > 0:
                    all_empty = False
                    break
            if all_empty:
                # All beams ended
                break

            # Flatten next_beam_hyps_list => beam_hyps again for next iteration
            beam_hyps = []
            for batch_i in range(batch_size):
                beam_hyps.extend(next_beam_hyps_list[batch_i])

        # 5) End of the main loop: collect leftover beams that haven't ended
        for cand in beam_hyps:
            batch_i = cand["batch_idx"]
            finished[batch_i].append(cand)

        # 6) For each example in the batch, pick the best final candidate by "combined_score" if available,
        #    else fallback to "lm_log_prob".
        outputs = []
        for batch_i in range(batch_size):
            # if somehow no finished => don't panic, just use whatever is in 'finished' 
            if not finished[batch_i]:
                # This means we have no ended or leftover => fallback
                outputs.append(torch.tensor([start_token_id], device=device))
                continue

            for f in finished[batch_i]:
                if "combined_score" not in f:
                    seq_len = len(f["tokens"])
                    lp = (5 + seq_len) / 6.0
                    f["combined_score"] = f["lm_log_prob"] / (lp ** length_penalty)

            # pick best final
            finished[batch_i].sort(key=lambda c: c["combined_score"], reverse=True)
            best = finished[batch_i][0]["tokens"]
            outputs.append(torch.tensor(best, device=device))

        # 7) Pad all outputs in this batch to same length & return a single tensor
        max_len = max(len(seq) for seq in outputs)
        padded = []
        for seq in outputs:
            needed = max_len - seq.size(0)
            if needed > 0:
                seq = torch.cat([seq, seq.new_full((needed,), self.tokenizer.pad_token_id)])
            padded.append(seq.unsqueeze(0))
        # shape => [B, max_len]
        final_outputs = torch.cat(padded, dim=0)

        return final_outputs

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

        device = self.device

        # 1) First pass
        best_output_ids = self._guided_beam_search_single_pass(
            inputs=inputs,
            generation_kwargs=generation_kwargs,
            checkpoint_interval=checkpoint_interval,
            beam_size=beam_size,
            alpha=alpha,
            beta=beta,
            length_penalty=length_penalty,
        )

        if num_passes <= 1:
            return best_output_ids

        # 2) For each additional pass, re-embed best final output & feed as "frozen_embeddings"
        for pass_idx in range(1, num_passes):
            # decode tokens => text
            txts = self.tokenizer.batch_decode(best_output_ids, skip_special_tokens=True)

            # re-embed final best text => shape (batch, embed_dim)
            new_embs = []
            for t in txts:
                emb_inp = self.embedder_tokenizer(t, return_tensors="pt").to(device)
                with torch.no_grad():
                    e = self.call_embedding_model(
                        input_ids=emb_inp["input_ids"],
                        attention_mask=emb_inp["attention_mask"]
                    )
                new_embs.append(e)
            new_embs = torch.cat(new_embs, dim=0)  # (batch, embed_dim)

            # feed that in as "frozen_embeddings" for the next pass
            new_inputs = dict(inputs)
            new_inputs["frozen_embeddings"] = new_embs

            # 3) guided beam search again
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
