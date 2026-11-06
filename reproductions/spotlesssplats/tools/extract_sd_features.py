"""Stable Diffusion 2.1 feature extraction of SpotLessSplats (official
examples/datasets/sd_feature_extraction.ipynb, itself DIFT, Tang et al. 2023): image resized to
800 x 800 (PIL default), x in [-1, 1], VAE latent sample * scaling factor, DDIM add_noise at
t = 261, empty prompt, UNet up_blocks[1] output averaged over an ensemble of 4 noise draws ->
[1280, 50, 50] float32 saved as <out>/<stem>.npy.

The notebook's MyUNet2DConditionModel (deform=None) is the stock UNet forward that also returns
up-block outputs; a forward hook on up_blocks[1] captures the same tensor.

Weights: ~/models/sd21 -> /bigdata/sejain/models/sd21 (sd2-community/stable-diffusion-2-1 @
bb21548, fp32 safetensors, sha256 in SHA256SUMS == HF LFS oids). diffusers 0.30.3 /
transformers 4.44.2 from ~/models/sd21base/pylib (PYTHONPATH fallback below).

    tools/ssenv python -I extract_sd_features.py --images DIR --out DIR [--names a.JPG b.JPG]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PYLIB = "/home/sejain/models/sd21base/pylib"
if PYLIB not in sys.path:
    sys.path.insert(0, PYLIB)  # its huggingface_hub 0.24 must shadow the env 2.x
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402


class Featurizer:
    def __init__(self, sd="/home/sejain/models/sd21", device="cuda"):
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(sd, subfolder="vae").to(device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(sd, subfolder="unet").to(device).eval()
        self.unet.set_attention_slice("auto")
        tok = CLIPTokenizer.from_pretrained(sd, subfolder="tokenizer")
        enc = CLIPTextModel.from_pretrained(sd, subfolder="text_encoder").to(device).eval()
        self.scheduler = DDIMScheduler.from_pretrained(sd, subfolder="scheduler")
        self.scheduler.set_timesteps(50)
        ids = tok([""], padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            self.prompt = enc(ids)[0]  # StableDiffusionPipeline.encode_prompt (clip_skip None)
        del enc
        self.captured = {}
        self.unet.up_blocks[1].register_forward_hook(
            lambda m, i, o: self.captured.__setitem__("ft", o.detach()))

    @torch.no_grad()
    def __call__(self, image, t=261, ensemble=4, size=800):
        img = image.convert("RGB").resize((size, size))
        x = (torch.tensor(np.array(img)) / 255.0 - 0.5) * 2
        x = x.permute(2, 0, 1)[None].repeat(ensemble, 1, 1, 1).to(self.device)
        lat = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        tt = torch.tensor(t, dtype=torch.long, device=self.device)
        noisy = self.scheduler.add_noise(lat, torch.randn_like(lat), tt)
        self.unet(noisy, tt, encoder_hidden_states=self.prompt.repeat(ensemble, 1, 1))
        return self.captured["ft"].mean(0).float().cpu().numpy()  # [1280, 50, 50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--compare", help="directory with reference .npy features (parity check)")
    a = ap.parse_args()
    feat = Featurizer()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.names or sorted(p.name for p in Path(a.images).iterdir())
    for n in names:
        f = feat(Image.open(Path(a.images) / n))
        np.save(out / f"{Path(n).stem}.npy", f)
        if a.compare:
            ref = np.load(Path(a.compare) / f"{Path(n).stem}.npy")
            ref = ref.reshape(f.shape)
            cos = (f * ref).sum(0) / (np.linalg.norm(f, axis=0) * np.linalg.norm(ref, axis=0))
            print(n, "shape", f.shape, "per-pixel cosine mean %.4f min %.4f" % (cos.mean(), cos.min()),
                  "rel L2 %.4f" % (np.linalg.norm(f - ref) / np.linalg.norm(ref)), flush=True)


if __name__ == "__main__":
    main()
