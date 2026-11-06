"""GS-IR renderer on gsplat.

Stage 1 (initial stage, no decomposition bank yet):
  pass A  gsplat RGB+ED with SH colours, exactly the baseline renderer (white background);
  pass B  (training only) Gaussian normals blended with the same weights but **detached
          geometry**: the official backward (backward.cu renderCUDA) sends rendered-normal
          gradients to the per-Gaussian normals only, never to alpha / means / covariances.
          normal = normalize(sum w_i n_i) where nonzero (gaussian_renderer/__init__.py).
  normal_from_depth = depthmapToNormalCUDA of the 3x3 median-filtered alpha-normalised depth
  (Eq. 7, linear interpolation of the Gaussian depths), no gradient.
Stage 2 (decomposition): one gsplat pass with detached geometry blending
  [albedo, roughness, metallic, normal] (sigmoid / normalised) + ED depth, then the split-sum
  PBR shader with baked occlusion and irradiance volumes (pbr/shade.py, train.py).
  Training (grad enabled) follows train.py: roughness r * 0.96 + 0.04, normals as rendered.
  Evaluation (no grad) follows render.py: inference=True (roughness += final transmittance),
  pad_normal=True (normal * O + (1 - O) * (0, 0, 1), O thresholded at 0.004 / 0.996), no
  roughness remap. Pixels with an all-zero rendered normal are black.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization

from . import bank as bank_mod
from . import geometry, light
from .volumes import query_irradiance, recon_occlusion

CAPABILITIES = (
    "gradient.screen.signed",
    "render.depth.expected_z",
    "render.normal.world",
    "render.normal_from_depth.world",
    "render.pbr.gbuffer",
    "render.rgb",
)
FIELDS = ("gsir_normal", "gsir_albedo", "gsir_roughness", "gsir_metallic")
DEFAULTS = {
    "rough_min": 0.04,
    "rough_max": 1.0,
    "metallic": True,
    "indirect": True,
    "occ_samples": 256,
    "eval_inference": True,
    "eval_pad_normal": True,
    "eval_rough_remap": False,
}


def _normalize_nonzero(x):
    n = x.norm(dim=-1, keepdim=True)
    return torch.where(n > 0, x / n.clamp_min(1e-30), x)


class GSIRRenderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS) - {"bank"}
        if unknown:
            raise ValueError(f"Unknown gsir-renderer options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **{k: v for k, v in spec.config.items() if k != "bank"}}
        self.bank_cfg = dict(spec.config.get("bank", {}))
        self.capabilities = set(CAPABILITIES)
        self.bank = None  # explicit bank (probes); otherwise bank.resolve()

    def active_bank(self, params):
        if self.bank is not None:
            return self.bank
        return bank_mod.resolve(params, self.bank_cfg)

    def render(self, params, camera, sh_degree=3):
        bank = self.active_bank(params)
        if bank is not None and bank.stage == 2:
            return self.render_pbr(params, camera, bank)
        return self.render_initial(params, camera, sh_degree)

    # ------------------------------------------------------------------ stage 1
    def _raster(self, params, camera, colors, sh_degree, mode, detach):
        g = (lambda t: t.detach()) if detach else (lambda t: t)
        return rasterization(
            means=g(params["means"]), quats=g(params["quats"]), scales=g(params["scales"]).exp(),
            opacities=g(params["opacities"]).sigmoid(), colors=colors,
            viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=camera["width"],
            height=camera["height"], sh_degree=sh_degree, packed=False, render_mode=mode,
        )

    def render_initial(self, params, camera, sh_degree):
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        out, alpha, info = self._raster(params, camera, colors, sh_degree, "RGB+ED", False)
        result = {
            "rgb": out[0, ..., :3] + (1 - alpha[0]),
            "depth": out[0, ..., 3:4],
            "alpha": alpha[0],
            "info": info,
            "gsir_stage": 1,
        }
        if torch.is_grad_enabled() and "gsir_normal" in params:
            nrm = F.normalize(params["gsir_normal"], dim=-1)
            acc, _, _ = self._raster(params, camera, nrm, None, "RGB", True)
            result["normal"] = _normalize_nonzero(acc[0])
            with torch.no_grad():
                c2w = torch.linalg.inv(camera["viewmat"])
                result["normal_from_depth"] = geometry.normal_from_depth(
                    result["depth"][..., 0].detach(), camera["K"], c2w[:3, :3])
        return result

    # ------------------------------------------------------------------ stage 2
    def gbuffer(self, params, camera):
        feats = torch.cat((params["gsir_albedo"].sigmoid(), params["gsir_roughness"].sigmoid(),
                           params["gsir_metallic"].sigmoid(),
                           F.normalize(params["gsir_normal"], dim=-1)), dim=-1)
        out, alpha, info = self._raster(params, camera, feats, None, "RGB+ED", True)
        o = out[0]
        return {"albedo": o[..., 0:3], "roughness": o[..., 3:4], "metallic": o[..., 4:5],
                "normal_acc": o[..., 5:8], "depth": o[..., 8:9], "alpha": alpha[0], "info": info}

    def render_pbr(self, params, camera, bank):
        training = torch.is_grad_enabled()
        c = self.cfg
        gb = self.gbuffer(params, camera)
        alpha = gb["alpha"]
        normal_acc = gb["normal_acc"]
        mask = (normal_acc != 0).all(dim=-1, keepdim=True)
        roughness = gb["roughness"]
        if training:
            roughness = roughness * (c["rough_max"] - c["rough_min"]) + c["rough_min"]
        else:
            if c["eval_inference"]:
                roughness = roughness + (1.0 - alpha)
            if c["eval_rough_remap"]:
                roughness = roughness * (c["rough_max"] - c["rough_min"]) + c["rough_min"]
            if c["eval_pad_normal"]:
                o = torch.where(alpha < 0.004, torch.zeros_like(alpha), alpha)
                o = torch.where(o > 1.0 - 0.004, torch.ones_like(o), o)
                bg = torch.tensor([0.0, 0.0, 1.0], device=o.device)
                normal_acc = normal_acc * o + (1.0 - o) * bg
        normal = _normalize_nonzero(normal_acc)
        depth = gb["depth"]
        view_dirs, points, _ = geometry.view_geometry(camera, depth)
        h, w = depth.shape[:2]
        if c["indirect"] and bank.indirect:
            b = bank.occ_bound
            pts = points.reshape(-1, 3).clamp(-b, b).contiguous()
            nrm = normal.reshape(-1, 3).detach().contiguous()
            with torch.no_grad():
                occlusion = recon_occlusion(bank.occ_coeffs, bank.occ_ids, b, pts, nrm,
                                            bank.occ_bands, int(c["occ_samples"])).reshape(h, w, 1)
            irradiance = query_irradiance(bank.irradiance, bank.irr_aabb, pts.detach(),
                                          nrm).reshape(h, w, -1)
        else:
            occlusion = torch.ones_like(roughness)
            irradiance = torch.zeros_like(roughness)
        spec, diffuse = bank.mips(training)
        shaded = light.pbr_shading(
            spec, diffuse, light.brdf_lut(depth.device), normal.detach(), view_dirs,
            gb["albedo"], roughness, mask,
            metallic=gb["metallic"] if c["metallic"] else None,
            occlusion=occlusion, irradiance=irradiance,
        )
        return {
            "rgb": shaded["rgb"],
            "depth": depth,
            "alpha": alpha,
            "info": gb["info"],
            "normal": normal,
            "gsir_stage": 2,
            "gbuffer": {"albedo": gb["albedo"], "roughness": roughness,
                        "metallic": gb["metallic"], "mask": mask, "occlusion": occlusion,
                        "irradiance": irradiance, "diffuse_rgb": shaded["diffuse_rgb"],
                        "specular_rgb": shaded["specular_rgb"]},
        }


def create_renderer(spec):
    return GSIRRenderer(spec)
