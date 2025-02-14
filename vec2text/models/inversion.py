import copy
import logging
import weakref  # add weakref for parent_model references

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sentence_transformers import SentenceTransformer
from transformers.modeling_outputs import Seq2SeqLMOutput

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

def is_main_process():
    """Helper to check if we're on rank 0 when using torchrun --nproc_per_node."""
    return int(os.environ.get("RANK", "0")) == 0

def debug_print(*args, **kwargs):
    """
    Print only on the main process, to avoid clutter from multiple GPUs.
    Flush output so we see it right away.
    """
    if is_main_process():
        print(*args, **kwargs, flush=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [GUIDED DIFFUSION ADDED]
# Full submodule implementing advanced diffusion with optional guidance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CosineScheduler:
    """Cosine schedule for betas over T timesteps."""
    def __init__(self, timesteps=50, beta_start=1e-4, beta_end=0.02):
        import math
        self.timesteps = timesteps
        betas = []
        for i in range(timesteps):
            f = i / (timesteps - 1)
            cos_val = 0.5 * (1 - math.cos(math.pi * f))
            beta = beta_start + cos_val * (beta_end - beta_start)
            betas.append(beta)
        self.betas = torch.tensor(betas, dtype=torch.float32)


class SelfConditionedDenoiser(nn.Module):
    """
    A basic Transformer-based denoiser that self-conditions on previous step's predicted x0.
    """
    def __init__(self, latent_dim=256, hidden_dim=1024, time_embed_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # We'll combine (z_t, z_hat) => (B,L,2*latent_dim)
        self.input_proj = nn.Linear(latent_dim*2, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, z_hat, t):
        """
        z_t, z_hat: shape (B,L,latent_dim) typically L=1 for single-vector latents
        t: shape (B,) integer timesteps
        """
        if z_hat is None:
            z_hat = torch.zeros_like(z_t)
        x = torch.cat([z_t, z_hat], dim=-1)  # (B,L,2*latent_dim)
        h = self.input_proj(x)

        # time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1).float())  # (B, time_embed_dim)
        t_emb = t_emb.unsqueeze(1).expand(h.size(0), h.size(1), t_emb.size(-1))
        if t_emb.size(-1) < h.size(-1):
            pad_dim = h.size(-1) - t_emb.size(-1)
            t_emb = F.pad(t_emb, (0, pad_dim), "constant", 0)
        h = h + t_emb

        h = self.transformer(h)
        out = self.out_proj(h)   # => (B,L,latent_dim)
        return out


class GuidedDiffusion(nn.Module):
    """
    Full diffusion submodule with:
      - Cosine schedule
      - Self-conditioning
      - Anchor loss
      - Optional guidance calls to embedding API (both training & inference)
      - Finite-difference approximation if black-box embedding
      - Uses a weak reference to the parent model to avoid recursion.
    """
    def __init__(
        self,
        parent_model,  # a strong ref is replaced by a weak ref below
        latent_dim=256,
        timesteps=50,
        anchor_weight=0.01,
        guidance_mode="cosine",  # or "mse"
        guidance_finite_diff_eps=1e-3,
    ):
        super().__init__()

        # Store a WEAK reference to avoid cyc recursion:
        self._parent_ref = weakref.ref(parent_model)

        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.anchor_weight = anchor_weight

        # guidance method
        self.guidance_mode = guidance_mode
        self.guidance_finite_diff_eps = guidance_finite_diff_eps

        # create a cosine schedule
        sched = CosineScheduler(timesteps, beta_start=1e-4, beta_end=0.02)
        self.register_buffer("betas", sched.betas)

        alphas = 1.0 - sched.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Defensive clamp to [0,1]
        alphas_cumprod = torch.clamp(alphas_cumprod, 0.0, 1.0)

        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # define the denoiser
        self.denoiser = SelfConditionedDenoiser(
            latent_dim=latent_dim,
            hidden_dim=4*latent_dim,
            time_embed_dim=256,
            num_layers=2,
            num_heads=4,
        )

    def _parent(self):
        """
        Utility to get the parent model (or None if it's gone).
        """
        return self._parent_ref()

    def q_sample(self, z0, t, noise=None):
        """
        Forward diffuse: z_t = sqrt(alpha_bar[t]) * z0 + sqrt(1 - alpha_bar[t]) * noise
        """
        if noise is None:
            noise = torch.randn_like(z0)
        alpha_bar = self.alphas_cumprod[t].view(-1,1,1)  # in [0..1], ideally
        return alpha_bar.sqrt() * z0 + (1 - alpha_bar).sqrt() * noise

    def forward(
        self,
        z0,
        t=None,
        z_hat=None,
        training_guidance_scale=0.0,
        training_guidance_freq=99999999,
    ):
        """
        Called in training. We:
          1) pick random t
          2) add noise
          3) predict noise
          4) anchor => x0_pred near z0
          5) optionally do partial decode & guidance if training_guidance_scale>0 
             and we haven't done it too frequently
        """
        B, L, D = z0.shape
        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=z0.device)

        # step 1: forward diffuse
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise=noise)

        if z_hat is None:
            z_hat = torch.zeros_like(z_t)

        # step 2: predict noise
        pred_noise = self.denoiser(z_t, z_hat, t)
        noise_mse = F.mse_loss(pred_noise, noise)

        # step 3: anchor
        alpha_bar = self.alphas_cumprod[t].view(-1,1,1)
        x0_pred = (z_t - (1 - alpha_bar).sqrt() * pred_noise) / (alpha_bar.sqrt() + 1e-7)
        anchor_loss = F.mse_loss(x0_pred, z0)

        total_loss = noise_mse + self.anchor_weight * anchor_loss

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # [TRAINING-TIME GUIDANCE]
        if (training_guidance_scale > 0.0) and (training_guidance_freq < 99999999):
            x0_pred = self.apply_embedding_guidance_training(
                x0_pred,
                z0,
                training_guidance_scale,
            )
            guidance_loss = self._compute_guidance_loss(x0_pred, z0)
            total_loss = total_loss + guidance_loss

        return total_loss
    
    def apply_embedding_guidance_training(
        self,
        x0_pred: torch.Tensor,
        z0: torch.Tensor, 
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        partial decode => embed => measure alignment with the original embedding 
        or some reference. We'll assume we want to match the *same* embedding as z0.
        """
        pm = self._parent()
        if pm is None:
            # parent is gone, skip
            return x0_pred

        device = x0_pred.device
        B, L, D = x0_pred.shape
        if L>1:
            lat_mean = x0_pred.mean(dim=1, keepdim=True)
        else:
            lat_mean = x0_pred

        # step A: decode => text
        texts = pm._decode_latent_to_text(lat_mean)

        # step B: embed
        new_embs = []
        for text in texts:
            tokenized = pm.embedder_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
            emb = pm.call_embedding_model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            )
            new_embs.append(emb)
        new_embs = torch.cat(new_embs, dim=0)  # (B, embed_dim)

        if not hasattr(pm, "_training_target_emb"):
            # can't do alignment
            return x0_pred  
        target_embedding = pm._training_target_emb

        # measure alignment => e.g. cos similarity
        if self.guidance_mode == "cosine":
            cos_sim = F.cosine_similarity(new_embs, target_embedding, dim=-1) 
            avg_score = cos_sim.mean()
        else:
            # mse
            mse_val = F.mse_loss(new_embs, target_embedding, reduction="none").mean(dim=-1)
            avg_score = -mse_val.mean()

        # step D: naive finite difference for a random subset of dims
        partial_dims = min(10, lat_mean.numel())
        dim_list = torch.randperm(lat_mean.numel(), device=device)[:partial_dims]
        best_update = torch.zeros_like(lat_mean)
        eps = self.guidance_finite_diff_eps

        old_flat = lat_mean.flatten()
        for d in dim_list:
            old_val = old_flat[d].item()
            old_flat[d] = old_val + eps
            lat_perturbed = old_flat.view(lat_mean.shape)

            texts_pert = pm._decode_latent_to_text(lat_perturbed)
            new_embs_pert = []
            for txt in texts_pert:
                tkn = pm.embedder_tokenizer([txt], return_tensors="pt").to(device)
                emb_pert = pm.call_embedding_model(
                    input_ids=tkn["input_ids"],
                    attention_mask=tkn["attention_mask"],
                )
                new_embs_pert.append(emb_pert)
            new_embs_pert = torch.cat(new_embs_pert, dim=0)

            if self.guidance_mode == "cosine":
                new_score = F.cosine_similarity(new_embs_pert, target_embedding, dim=-1).mean().item()
            else:
                new_score = -F.mse_loss(new_embs_pert, target_embedding).item()

            grad_approx = (new_score - avg_score.item()) / eps
            # revert
            old_flat[d] = old_val

            best_update_flat = best_update.flatten()
            best_update_flat[d] = grad_approx
            best_update = best_update_flat.view(best_update.shape)

        lat_new = lat_mean + guidance_scale * best_update
        if L > 1:
            lat_new = lat_new.expand(B, L, D)
        return lat_new

    def _compute_guidance_loss(self, x0_pred, z0):
        """
        We define a 'guidance_loss' to incorporate partial decode alignment
        into the training loss.
        """
        pm = self._parent()
        if pm is None or not hasattr(pm, "_training_target_emb"):
            return torch.tensor(0.0, device=x0_pred.device)

        target_emb = pm._training_target_emb
        device = x0_pred.device

        B, L, D = x0_pred.shape
        lat_mean = x0_pred.mean(dim=1, keepdim=True)
        texts = pm._decode_latent_to_text(lat_mean)
        new_embs = []
        for text in texts:
            tkn = pm.embedder_tokenizer([text], return_tensors="pt").to(device)
            emb = pm.call_embedding_model(
                input_ids=tkn["input_ids"],
                attention_mask=tkn["attention_mask"]
            )
            new_embs.append(emb)
        new_embs = torch.cat(new_embs, dim=0)

        cos_sim = F.cosine_similarity(new_embs, target_emb, dim=-1).mean()
        # define negative cos => guidance_loss
        guidance_loss = -cos_sim
        return guidance_loss

    def p_sample_loop(
        self,
        x_T,
        steps=None,
        inference_guidance_scale=1.0,
        inference_guidance_freq=1,
        target_embedding: Optional[torch.Tensor] = None,
    ):
        if steps is None:
            steps = self.timesteps
        z_hat = torch.zeros_like(x_T)
        x_t = x_T

        for i in reversed(range(steps)):
            t_tensor = torch.full((x_t.size(0),), i, device=x_t.device, dtype=torch.long)

            # (A) Print stats for x_t before denoising
            if i % 5 == 0:  # e.g. print every 5 steps
                print(f"[p_sample_loop] step={i}, x_t min/mean/max:",
                    x_t.min().item(),
                    x_t.mean().item(),
                    x_t.max().item(), flush=True)

            # 1: Denoise
            pred_noise = self.denoiser(x_t, z_hat, t_tensor)
            
            print(f"[p_sample_loop] alphas_cumprod: {self.alphas_cumprod}")

            alpha_bar = self.alphas_cumprod[t_tensor].view(-1,1,1)
            x0_pred = (x_t - (1 - alpha_bar).sqrt() * pred_noise) / (alpha_bar.sqrt() + 1e-7)

            # (B) Print stats for x0_pred
            if i % 5 == 0:
                print(f"[p_sample_loop] step={i}, x0_pred min/mean/max:",
                    x0_pred.min().item(),
                    x0_pred.mean().item(),
                    x0_pred.max().item(),
                    flush=True)

            # 2: Guidance
            if (inference_guidance_scale>0.0) and (i % inference_guidance_freq == 0) and (target_embedding is not None):
                x0_pred = self.apply_embedding_guidance(
                    x0_pred,
                    inference_guidance_scale,
                    target_embedding=target_embedding,
                )

            z_hat = x0_pred.detach()

            # 3: Sample posterior (unless i==0)
            if i > 0:
                beta_t = self.betas[t_tensor].view(-1,1,1)
                alpha_bar_prev = self.alphas_cumprod[t_tensor - 1].view(-1,1,1)
                denom = (1 - alpha_bar).sqrt() + 1e-7

                mean = (
                    x0_pred * alpha_bar_prev.sqrt()
                    + (x_t - x0_pred * alpha_bar.sqrt()) * (1 - alpha_bar_prev).sqrt()
                ) / denom

                var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar)
                noise = torch.randn_like(x_t)

                # (C) Print stats for mean, var if you suspect negativity or huge values:
                if i % 5 == 0:
                    print(f"[p_sample_loop] step={i}, mean min/mean/max:",
                        mean.min().item(),
                        mean.mean().item(),
                        mean.max().item(),
                        flush=True)
                    print(f"[p_sample_loop] step={i}, var min/mean/max:",
                        var.min().item(),
                        var.mean().item(),
                        var.max().item(),
                        flush=True)
                
                x_t = mean + var.sqrt() * noise

        return x_t


    def apply_embedding_guidance(
        self,
        x0_pred: torch.Tensor,
        guidance_scale: float,
        target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        1) decode partial latent => text
        2) embed => measure alignment
        3) do naive finite-diff => nudge x0_pred
        """
        pm = self._parent()
        if pm is None:
            return x0_pred

        device = x0_pred.device
        B, L, D = x0_pred.shape

        if L > 1:
            lat_mean = x0_pred.mean(dim=1, keepdim=True)
        else:
            lat_mean = x0_pred

        texts = pm._decode_latent_to_text(lat_mean)

        new_embs = []
        for text in texts:
            tokenized = pm.embedder_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
            emb = pm.call_embedding_model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"]
            )
            new_embs.append(emb)
        new_embs = torch.cat(new_embs, dim=0)

        if self.guidance_mode == "cosine":
            cos_sim = F.cosine_similarity(new_embs, target_embedding, dim=-1).mean()
            avg_score = cos_sim.item()
        else:
            # negative MSE
            mse_val = F.mse_loss(new_embs, target_embedding)
            avg_score = -mse_val.item()

        # do finite difference
        partial_dims = min(10, lat_mean.numel())
        dim_list = torch.randperm(lat_mean.numel(), device=device)[:partial_dims]
        best_update = torch.zeros_like(lat_mean)
        eps = self.guidance_finite_diff_eps

        lat_flat = lat_mean.flatten()
        for d in dim_list:
            old_val = lat_flat[d].item()
            lat_flat[d] = old_val + eps
            lat_perturbed = lat_flat.view(lat_mean.shape)

            texts_pert = pm._decode_latent_to_text(lat_perturbed)
            new_embs_pert = []
            for txt in texts_pert:
                tkn = pm.embedder_tokenizer([txt], return_tensors="pt").to(device)
                emb_pert = pm.call_embedding_model(
                    input_ids=tkn["input_ids"],
                    attention_mask=tkn["attention_mask"],
                )
                new_embs_pert.append(emb_pert)
            new_embs_pert = torch.cat(new_embs_pert, dim=0)

            if self.guidance_mode == "cosine":
                new_score = F.cosine_similarity(new_embs_pert, target_embedding, dim=-1).mean().item()
            else:
                new_score = -F.mse_loss(new_embs_pert, target_embedding).item()

            grad_approx = (new_score - avg_score) / eps
            lat_flat[d] = old_val

            best_update_flat = best_update.flatten()
            best_update_flat[d] = grad_approx
            best_update = best_update_flat.view(best_update.shape)

        lat_new = lat_mean + guidance_scale * best_update
        if L > 1:
            lat_new_exp = lat_new.expand(B, L, D)
        else:
            lat_new_exp = lat_new
        return lat_new_exp


@dataclass
class DiffusionSeq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Extends Seq2SeqLMOutput with extra fields for CE loss and diffusion loss.
    """
    ce_loss: Optional[torch.FloatTensor] = None
    diffusion_loss: Optional[torch.FloatTensor] = None


class InversionModel(transformers.PreTrainedModel):
    """A model that inverts embeddings to text, now with optional guided diffusion."""

    config_class = InversionConfig
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool
    tokenizer: transformers.PreTrainedTokenizer
    embedding_transform: nn.Module
    bottleneck_dim: int
    num_repeat_tokens: int
    embedder_dim: int
    embedder_no_grad: bool
    embedder_fake_with_zeros: bool
    embedding_transform_strategy: str
    use_frozen_embeddings_as_input: bool
    embedded_tokens: torch.Tensor
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

        self.encoder_decoder = encoder_decoder
        self.num_repeat_tokens = num_repeat_tokens

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        if embedder_model_api:
            # Hard-code e.g. OpenAI dimension
            self.embedder_dim = 1536
            bottleneck_dim = self.embedder_dim
        elif isinstance(embedder, SentenceTransformer):
            self.embedder_dim = embedder.get_sentence_embedding_dimension()
            bottleneck_dim = self.embedder_dim
        else:
            if hasattr(embedder.config, "text_config"):
                self.embedder_dim = embedder.config.text_config.hidden_size
            elif hasattr(embedder.config, "hidden_size"):
                self.embedder_dim = embedder.config.hidden_size
            else:
                raise ValueError(
                    f"could not determine hidden size for embedder {embedder} of type {type(embedder)}"
                )
            bottleneck_dim = self.embedder_dim

        self.embedder_no_grad = embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        self.bottleneck_dim = bottleneck_dim

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
        if self.embedder_no_grad:
            for param in self.embedder.parameters():
                param.requires_grad = False
            self.embedder.eval()

        self.embedder_tokenizer = embedder_tokenizer
        self.embedder_model_api = embedder_model_api
        self.embedder_fake_with_zeros = embedder_fake_with_zeros
        self.embedding_transform_strategy = "repeat"
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = vars(config).get("embedder_gaussian_noise_level", 0.0)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # [GUIDED DIFFUSION ADDED]
        self.use_diffusion = getattr(config, "use_diffusion", False)
        self.latent_dim = getattr(config, "latent_dim", 256)
        self.diffusion_timesteps = getattr(config, "diffusion_timesteps", 50)
        self.diffusion_anchor_weight = getattr(config, "diffusion_anchor_weight", 0.01)

        # Guidance parameters
        self.diffusion_training_guidance_scale = getattr(config, "diffusion_training_guidance_scale", 0.0)
        self.diffusion_training_guidance_freq = getattr(config, "diffusion_training_guidance_freq", 9999999999)

        self.diffusion_inference_guidance_scale = getattr(config, "diffusion_inference_guidance_scale", 0.0)
        self.diffusion_inference_guidance_freq = getattr(config, "diffusion_inference_guidance_freq", 9999999999)

        # possible other config fields for guidance mode (cosine vs. mse) or finite diff
        self.diffusion_guidance_mode = getattr(config, "diffusion_guidance_mode", "cosine")
        self.diffusion_guidance_finite_diff_eps = getattr(config, "diffusion_guidance_finite_diff_eps", 1e-3)

        # build submodule
        if self.use_diffusion:
            self.guided_diffusion = GuidedDiffusion(
                parent_model=self,
                latent_dim=self.latent_dim,
                timesteps=self.diffusion_timesteps,
                anchor_weight=self.diffusion_anchor_weight,
                guidance_mode=self.diffusion_guidance_mode,
                guidance_finite_diff_eps=self.diffusion_guidance_finite_diff_eps,
            )

    def _freeze_encoder(self):
        freeze_params(self.encoder_decoder.encoder)

    def _freeze_decoder(self):
        freeze_params(self.encoder_decoder.decoder)
        freeze_params(self.encoder_decoder.lm_head)

    def freeze(self, freeze_strategy: str):
        if freeze_strategy not in FREEZE_STRATEGIES:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")
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
                assert hasattr(outputs, "hidden_states"), "need output_hidden_states=True"
                hidden_state = outputs.hidden_states[self.embeddings_from_layer_n]
                embeddings = mean_pool(hidden_state, attention_mask)
            else:
                embeddings = mean_pool(outputs.last_hidden_state, attention_mask)
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
            B = input_ids.size(0)
            return torch.zeros((B, self.embedder_dim), dtype=torch.float32, device=self.embedder_device)
        elif self.embedder_model_api:
            # external API
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
            # local huggingface model
            model_output = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self._process_embedder_output(model_output, attention_mask)

        if self.training and self.noise_level > 0:
            embeddings += self.noise_level * torch.randn_like(embeddings)
        return embeddings

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (embedder_input_ids is None) and (frozen_embeddings is None):
            raise ValueError("Need either embedder_input_ids or frozen_embeddings")

        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
        else:
            embeddings = self.call_embedding_model(embedder_input_ids, embedder_attention_mask)

        # transform to repeated T5 input
        repeated_embeddings = self.embedding_transform(embeddings)
        B = repeated_embeddings.size(0)
        hidden_dim = self.encoder_decoder.config.hidden_size
        repeated_embeddings = repeated_embeddings.view(B, self.num_repeat_tokens, hidden_dim)
        attention_mask = torch.ones((B, self.num_repeat_tokens), device=repeated_embeddings.device)

        return repeated_embeddings, attention_mask

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Perform diffusion-based generation but return a typical T5 sequence shape [batch, seq_len].
        """
        if not self.use_diffusion:
            debug_print("---Not using diffusion---")
            return self._naive_generate(inputs, generation_kwargs)

        # 1) embed & project => shape [B, orig_seq_len, hidden_dim]
        inputs_embeds, _ = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )
        debug_print(f'--inputs_embeds.shape: {inputs_embeds.shape}')
        lat_mean = inputs_embeds.mean(dim=1, keepdim=True)

        # 2) partial noising => x_T (use q_sample to keep formula consistent)
        if not hasattr(self, "_diffusion_compress"):
            self._diffusion_compress = nn.Linear(lat_mean.size(-1), self.latent_dim).to(self.device)
        z0 = self._diffusion_compress(lat_mean)
        print(f"--generate() :: z0: {z0}")
        print(f"--generate() :: z0 min/mean/max: {z0.min()}, {z0.mean()}, {z0.max()}")
        T = self.diffusion_timesteps
        noise = torch.randn_like(z0)

        # Clean approach: just call q_sample with t = T-1
        xT = self.guided_diffusion.q_sample(z0, t=T-1, noise=noise)

        # 3) reverse diffusion => final latent x0
        refine_steps = generation_kwargs.pop("refine_steps", T)
        target_emb = inputs.get("target_embedding", None)
        x0 = self.guided_diffusion.p_sample_loop(
            x_T=xT,
            steps=refine_steps,
            inference_guidance_scale=self.diffusion_inference_guidance_scale,
            inference_guidance_freq=self.diffusion_inference_guidance_freq,
            target_embedding=target_emb,
        )

        # 4) expand x0 => shape [B, orig_seq_len, hidden_dim]
        if not hasattr(self, "_diffusion_expand"):
            self._diffusion_expand = nn.Linear(self.latent_dim, lat_mean.size(-1)).to(self.device)
        expanded = self._diffusion_expand(x0).expand(-1, inputs_embeds.size(1), -1)
        debug_print(f"---expanded: {expanded}---")

        # 5) T5 generate => typical shape [batch_size, dec_len]
        attention_mask = torch.ones(
            (expanded.size(0), expanded.size(1)),
            dtype=torch.long,
            device=expanded.device,
        )
        debug_print(f"---attention_mask: {attention_mask}---")
        out_ids = self.encoder_decoder.generate(
            inputs_embeds=expanded,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        debug_print(f"---out_ids: {out_ids}---")

        # 6) (Optional) pad to a fixed max_length
        max_len = generation_kwargs.get("max_length", 128)
        if out_ids.size(1) < max_len:
            pad_amount = max_len - out_ids.size(1)
            pad_tokens = torch.full(
                (out_ids.size(0), pad_amount),
                self.tokenizer.pad_token_id,
                dtype=out_ids.dtype,
                device=out_ids.device,
            )
            out_ids = torch.cat([out_ids, pad_tokens], dim=1)

        debug_print(f"---out_ids FINAL: {out_ids}---")
        return out_ids

    def _naive_generate(self, inputs: Dict[str, torch.Tensor], generation_kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        debug_print("--Doing _naive_generate--")
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
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> DiffusionSeq2SeqLMOutput:
        debug_print("---In forward()---")
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=frozen_embeddings,
        )
        debug_print(f"---forward() :: inputs_embeds.shape: {inputs_embeds.shape}---")
        debug_print(f"---forward() :: attention_mask.shape: {attention_mask.shape}")
        seq2seq_outputs = self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )

        ce_loss = seq2seq_outputs.loss
        diffusion_loss = None
        total_loss = ce_loss

        if self.use_diffusion and self.training:
            debug_print(f"---forward() :: Using diffusion and training---")
            lat_mean = inputs_embeds.mean(dim=1, keepdim=True)
            if not hasattr(self, "_diffusion_compress"):
                self._diffusion_compress = nn.Linear(lat_mean.size(-1), self.latent_dim).to(self.device)
            z0 = self._diffusion_compress(lat_mean)

            diff_loss = self.guided_diffusion(
                z0,
                t=None,
                z_hat=None,
                training_guidance_scale=self.diffusion_training_guidance_scale,
                training_guidance_freq=self.diffusion_training_guidance_freq,
            )
            diffusion_loss = diff_loss
            total_loss = ce_loss + 0.1 * diffusion_loss

        return DiffusionSeq2SeqLMOutput(
            loss=total_loss,
            logits=seq2seq_outputs.logits,
            past_key_values=seq2seq_outputs.past_key_values,
            decoder_hidden_states=seq2seq_outputs.decoder_hidden_states,
            decoder_attentions=seq2seq_outputs.decoder_attentions,
            cross_attentions=seq2seq_outputs.cross_attentions,
            encoder_last_hidden_state=seq2seq_outputs.encoder_last_hidden_state,
            encoder_hidden_states=seq2seq_outputs.encoder_hidden_states,
            encoder_attentions=seq2seq_outputs.encoder_attentions,
            ce_loss=ce_loss,
            diffusion_loss=diffusion_loss,
        )

    def _decode_latent_to_text(self, latent: torch.Tensor) -> list:
        """
        Used by apply_embedding_guidance in p_sample_loop to do partial decode 
        from a single-step latent. We do a small linear -> T5 -> decode.
        """
        B, L, D = latent.shape
        if not hasattr(self, "_diffusion_expand"):
            self._diffusion_expand = nn.Linear(D, self.encoder_decoder.config.hidden_size).to(self.device)

        hidden = self._diffusion_expand(latent)  # => (B,L, hidden_size)
        attn_mask = torch.ones((B,L), dtype=torch.long, device=latent.device)
        out_ids = self.encoder_decoder.generate(
            inputs_embeds=hidden,
            attention_mask=attn_mask,
            max_new_tokens=40,
            do_sample=True,
            num_beams=1,
            top_p=0.9,
            top_k=50,
        )
        texts = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return texts
