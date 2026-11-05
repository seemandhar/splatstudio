"""Score Distillation Sampling with Stable Diffusion 2.1-base (paper §3.3.1, Eq. 7-8).

Port of the official guidance/sd_utils.py StableDiffusion (DreamGaussian-derived): fp16,
DDIM scheduler, t_range [0.02, 0.98], empty prompts for both CFG branches,
``train_step``: bilinear 512x512 -> VAE encode (posterior sample x scaling factor, with grads)
-> fixed t = round((1 - step_ratio) * 1000) clipped to [min_step, max_step] -> add noise ->
UNet on [noisy]*2 with [neg, pos] embeddings -> eps = uncond + scale (cond - uncond)
-> grad = (1 - alphas_cumprod[t]) (eps - noise) -> loss = 0.5 sum((lat - (lat - grad))^2) / B.

Weights: ~/models/sd21base (sd2-community/stable-diffusion-2-1-base, fp16 safetensors; the
stabilityai repository is no longer available). diffusers 0.30.3 / transformers 4.44.2 live in
~/models/sd21base/pylib so the splatstudio env is not modified.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_PINNED = {"diffusers", "transformers", "tokenizers", "huggingface_hub"}


def sds_target_loss(latents, noise_pred, noise, w):
    """The official SDS surrogate: d loss / d latents = w (noise_pred - noise)."""
    grad = torch.nan_to_num(w * (noise_pred - noise))
    target = (latents - grad).detach()
    return 0.5 * F.mse_loss(latents.float(), target.float(), reduction="sum") / latents.shape[0]


def fixed_timestep(step_ratio, num_train_timesteps=1000, t_range=(0.02, 0.98)):
    lo, hi = int(num_train_timesteps * t_range[0]), int(num_train_timesteps * t_range[1])
    return int(np.round((1 - step_ratio) * num_train_timesteps).clip(lo, hi))


class StableDiffusionGuidance:
    def __init__(self, model_dir, pylib, device, fp16=True, t_range=(0.02, 0.98), modules=None):
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        if modules is None:
            modules = self._load(os.path.expanduser(model_dir), os.path.expanduser(pylib))
        self.vae, self.unet, self.scheduler, self.embeddings = modules
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.alphas = self.scheduler.alphas_cumprod.to(device)

    def _load(self, model_dir, pylib):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Stable Diffusion 2.1-base weights not found in {model_dir}")
        if pylib and pylib not in sys.path:
            # pinned diffusers stack must win over the env's newer huggingface_hub
            sys.path.insert(0, pylib)
            for name in [m for m in sys.modules if m.split(".")[0] in _PINNED]:
                del sys.modules[name]
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer, logging

        logging.set_verbosity_error()
        kw = {"torch_dtype": self.dtype, "variant": "fp16" if self.dtype == torch.float16 else None}
        vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae", **kw).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="unet", **kw).to(
            self.device)
        tok = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
        text = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder", **kw).to(
            self.device)
        scheduler = DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler",
                                                  torch_dtype=self.dtype)
        for m in (vae, unet, text):
            m.requires_grad_(False)
            m.eval()
        with torch.no_grad():
            ids = tok([""], padding="max_length", max_length=tok.model_max_length,
                      return_tensors="pt").input_ids.to(self.device)
            e = text(ids)[0]
        embeddings = torch.cat([e, e], dim=0)  # get_text_embeds([""], [""]): [neg, pos]
        del text
        torch.cuda.empty_cache()
        return vae, unet, scheduler, embeddings

    def encode_imgs(self, imgs):
        posterior = self.vae.encode(2 * imgs - 1).latent_dist
        return posterior.sample() * self.vae.config.scaling_factor

    def train_step(self, pred_rgb, step_ratio=None, guidance_scale=100.0):
        B = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
        latents = self.encode_imgs(
            F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False))
        if step_ratio is not None:
            t = fixed_timestep(step_ratio, self.num_train_timesteps, self.t_range)
            t = torch.full((B,), t, dtype=torch.long, device=self.device)
        else:
            lo = int(self.num_train_timesteps * self.t_range[0])
            hi = int(self.num_train_timesteps * self.t_range[1])
            t = torch.randint(lo, hi + 1, (B,), dtype=torch.long, device=self.device)
        w = (1 - self.alphas[t]).view(B, 1, 1, 1)
        with torch.no_grad():
            noise = torch.randn_like(latents)
            noisy = self.scheduler.add_noise(latents, noise, t)
            pred = self.unet(torch.cat([noisy] * 2), torch.cat([t] * 2),
                             encoder_hidden_states=self.embeddings.repeat(B, 1, 1)).sample
            uncond, cond = pred.chunk(2)
            pred = uncond + guidance_scale * (cond - uncond)
        return sds_target_loss(latents, pred, noise, w)
