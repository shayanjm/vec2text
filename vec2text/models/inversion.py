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
        
        self.max_seq_length = config.max_seq_length
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

        self.log_sigma_ce = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cosine_embedding = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_mse_embedding = nn.Parameter(torch.tensor(0.0))

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
    ) -> torch.Tensor:
        """
        Override generate to use beam search with normalized metrics.
        """
        # Determine the device
        device = next(self.parameters()).device

        generation_kwargs = copy.copy(generation_kwargs)  # Make a copy to edit
        beam_width = 50
        max_length = self.max_seq_length
        embedding_check_interval = generation_kwargs.pop("embedding_check_interval", 5)

        # Extract learned uncertainty parameters
        sigma_ce2 = torch.exp(self.log_sigma_ce)
        sigma_cos2 = torch.exp(self.log_sigma_cosine_embedding)
        sigma_mse2 = torch.exp(self.log_sigma_mse_embedding)

        # Fetch target embeddings
        target_embedding = inputs["frozen_embeddings"]

        # Ensure bos_token_id is valid
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = self.tokenizer.cls_token_id
        if bos_token_id is None:
            bos_token_id = self.tokenizer.eos_token_id
        if bos_token_id is None:
            bos_token_id = self.tokenizer.pad_token_id
        if bos_token_id is None:
            raise ValueError("No valid bos_token_id found in the tokenizer.")

        # Initialize beams
        beams = [{
            "tokens": [bos_token_id],
            "log_prob_sum": 0.0,
            "ce": 0.0,  # Cross-entropy term
            "metrics": {},
            "score": 0.0  # Total score including all terms
        }]
        completed_beams = []

        # Ensure eos_token_id is valid
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.sep_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            raise ValueError("No valid eos_token_id found in the tokenizer.")

        for step in range(max_length):
            new_beams = []
            metric_values = {"cosine": [], "mse": [], "ce": []}  # Collect metrics for normalization

            for beam in beams:
                tokens = beam["tokens"]
                log_prob_sum = beam["log_prob_sum"]

                # Expand current beam
                input_ids = torch.tensor([tokens], device=device)
                outputs = self.encoder_decoder(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
                next_token_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Take top-k tokens for beam expansion
                top_k_log_probs, top_k_tokens = torch.topk(next_token_log_probs, beam_width, dim=-1)

                for log_prob, token in zip(top_k_log_probs.squeeze(), top_k_tokens.squeeze()):
                    new_tokens = tokens + [token.item()]
                    new_log_prob_sum = log_prob_sum + log_prob.item()
                    new_ce = -new_log_prob_sum  # Cross-entropy is negative log probability sum

                    metrics = {}
                    # Periodically compute embedding metrics
                    if step % embedding_check_interval == 0 or step == max_length - 1:
                        partial_sequence = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        pred_input_ids = self.tokenizer.encode(
                            partial_sequence,
                            return_tensors='pt',
                            add_special_tokens=True
                        ).to(device)
                        pred_attention_mask = torch.ones_like(pred_input_ids).to(device)
                        pred_embedding = self.call_embedding_model(
                            input_ids=pred_input_ids,
                            attention_mask=pred_attention_mask,
                        )

                        # Compute Cosine Similarity and MSE
                        cosine_sim = torch.nn.functional.cosine_similarity(
                            pred_embedding, target_embedding, dim=-1
                        ).mean().item()
                        mse = torch.nn.functional.mse_loss(pred_embedding, target_embedding).item()

                        # Save metrics for normalization
                        metrics["cosine"] = cosine_sim
                        metrics["mse"] = mse
                        metric_values["cosine"].append(cosine_sim)
                        metric_values["mse"].append(mse)

                    # Collect CE for normalization
                    metric_values["ce"].append(new_ce)

                    new_beams.append({
                        "tokens": new_tokens,
                        "log_prob_sum": new_log_prob_sum,
                        "ce": new_ce,
                        "metrics": metrics,
                        "score": 0.0  # Will be updated after normalization
                    })

            # Normalize metrics across beams
            for metric in ["cosine", "mse", "ce"]:
                values = metric_values[metric]
                if len(values) > 0:
                    min_val, max_val = min(values), max(values)
                    # Avoid division by zero if all values are the same
                    denom = max_val - min_val + 1e-8
                    for beam in new_beams:
                        if metric == "ce":
                            normalized_value = (beam["ce"] - min_val) / denom
                            beam["normalized_ce"] = normalized_value
                        elif metric in beam["metrics"]:
                            normalized_value = (beam["metrics"][metric] - min_val) / denom
                            beam["metrics"][metric] = normalized_value

            # Update scores with uncertainty weighting
            for beam in new_beams:
                total_score = 0.0

                # Cross-Entropy Term (normalized)
                total_score -= (1 / sigma_ce2) * beam["normalized_ce"]  # Minimize normalized CE

                # Cosine Similarity Term (normalized)
                if "cosine" in beam["metrics"]:
                    total_score += (1 / sigma_cos2) * beam["metrics"]["cosine"]  # Maximize normalized cosine similarity

                # MSE Term (normalized)
                if "mse" in beam["metrics"]:
                    total_score -= (1 / sigma_mse2) * beam["metrics"]["mse"]  # Minimize normalized MSE

                beam["score"] = total_score

            # Prune to top-k beams
            beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_width]

            # Check for completed sequences
            for beam in beams:
                if beam["tokens"][-1] == eos_token_id:
                    completed_beams.append(beam)

            if len(completed_beams) >= beam_width:
                break

        # If no completed beams, take the best from current beams
        if not completed_beams:
            completed_beams = beams

        # Return the best sequence
        best_beam = sorted(completed_beams, key=lambda x: x["score"], reverse=True)[0]
        best_sequence = best_beam["tokens"]
        return torch.tensor(best_sequence, device=device)

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
