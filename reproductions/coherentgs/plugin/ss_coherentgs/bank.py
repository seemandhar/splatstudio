"""Process-local global state of CoherentGS (the native runtime has no global learned modules).

The density component owns one `Bank` and registers it in ``ACTIVE``; the renderer and the
losses read it while training (grad enabled). It holds the per-pixel ray structure
(normalised image coordinates, camera-to-world), the aligned initial depth D_init, the depth
layer masks S, the flows / consistency masks, the two implicit decoders with their AdamW +
CyclicLR optimisers, and the step counter. Evaluation never needs it: every training render
writes the derived means / opacity logits back into ``params`` (see renderer.py).

Official references (github.com/avinashpaliwal/CoherentGS): scene/gaussian_model.py
(get_xyz, get_residual, get_learned_opac, set_scaling, update_initz_so, set_ref_camera),
train.py (warping_loss, coarse alignment loop).
"""

import math
import random

import torch
import torch.nn.functional as F

from .decoder import ImplicitDecoder, backwarp, posenc

ACTIVE = {"bank": None}

# Official capacity tables (gaussian_model.py set_ref_camera) and initial opacity (create_from_pcd).
DEPTH_NGF = {2: 10, 3: 15, 4: 18}
OPAC_NGF = {2: 6, 3: 10, 4: 12}
INIT_OPACITY = {2: 0.6, 3: 0.5, 4: 0.35}


def pixel_rays(K, height, width, device):
    """kornia depth_to_3d convention: integer pixel (u, v); xy = ((u-cx)/fx, (v-cy)/fy)."""
    v, u = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                          torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return torch.stack([x, y], -1).reshape(-1, 2)


def masked_channel_sum(raw, mask_norm, scale):
    """scale * chsum(tanh(raw) * S / (sum_c S + 1e-10)) -> [N*H*W, 1] (official get_residual)."""
    return scale * torch.sum(torch.tanh(raw) * mask_norm, dim=1, keepdim=True).reshape(-1, 1)


def tv(inp):
    """Official train.py tv(): mean |dx| + mean |dy| of a [C, H, W] tensor."""
    dx = inp[:, :, :-1] - inp[:, :, 1:]
    dy = inp[:, :-1, :] - inp[:, 1:, :]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))


class Bank:
    def __init__(self, viewmats, K, z_mono, dep_mask, flows, flow_w, cfg, device):
        self.cfg = dict(cfg)
        dev = torch.device(device)
        self.device = dev
        self.viewmats = torch.as_tensor(viewmats, dtype=torch.float32, device=dev)
        self.K = torch.as_tensor(K, dtype=torch.float32, device=dev)
        z_mono = torch.as_tensor(z_mono, dtype=torch.float32, device=dev)
        self.n, self.height, self.width = (int(s) for s in z_mono.shape)
        if self.n != int(cfg["n_views"]):
            raise ValueError(f"sidecar has {self.n} views, config n_views={cfg['n_views']}")
        self.hw = self.height * self.width
        self.c2w = torch.linalg.inv(self.viewmats)
        self.xy = pixel_rays(self.K, self.height, self.width, dev).repeat(self.n, 1)
        self.z_mono = z_mono.reshape(-1, 1)
        self.z = self.z_mono.clone()  # D_init (aligned after `align`)
        self.dep_mask = torch.as_tensor(dep_mask, device=dev).float()
        if self.dep_mask.shape[:2] != (self.n, int(cfg["mask_channels"])):
            raise ValueError(f"dep_mask shape {tuple(self.dep_mask.shape)} != (N, C, H, W)")
        self.mask_norm = self.dep_mask / (torch.sum(self.dep_mask, dim=1, keepdim=True) + 1e-10)
        flows = torch.as_tensor(flows, dtype=torch.float32, device=dev)
        flow_w = torch.as_tensor(flow_w, dtype=torch.float32, device=dev)
        thr = float(cfg["flow_mask_threshold"])
        # Official: mask01[mask01 <= 0.5] = 0; mask01[mask01 > 0.5] = 1 (binary consistency mask).
        self.flows = flows.permute(0, 1, 4, 2, 3).contiguous()  # [N, N, 2, H, W]
        self.flow_masks = (flow_w > thr).float()[:, :, None]  # [N, N, 1, H, W]
        self.rng = random.Random(int(cfg["seed"]))
        self.inp = posenc(self.n).to(dev)
        self.opac0_logit = math.log(cfg["init_opacity"] / (1 - cfg["init_opacity"]))
        self.align_scale = torch.ones(self.n, device=dev)
        self.align_offset = torch.zeros(self.n, device=dev)
        self.step = None
        self.decoder = self.decoder_opac = None

    # ------------------------------------------------------------ geometry
    def xyz(self, z):
        """g(z, p): world xyz of every pixel Gaussian at camera depth z [N*H*W, 1]."""
        cam = torch.cat((self.xy * z, z), dim=1).reshape(self.n, self.hw, 3)
        world = cam @ self.c2w[:, :3, :3].transpose(1, 2) + self.c2w[:, None, :3, 3]
        return world.reshape(-1, 3)

    def view_index(self, viewmat, tol=1e-4):
        err = (self.viewmats - viewmat.detach().float()[None]).abs().amax(dim=(1, 2))
        i = int(err.argmin())
        if float(err[i]) > tol:
            raise KeyError(f"CoherentGS: camera is not a training view (closest {float(err[i])})")
        return i

    # ------------------------------------------------------------ decoders
    def build_decoders(self, seed):
        cfg = self.cfg
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(int(seed))
            c = int(cfg["mask_channels"])
            self.decoder = ImplicitDecoder(self.height, self.width, ngf=int(cfg["ngf_depth"]),
                                           in_ch=20, out_ch=c).to(self.device)
            self.decoder_opac = ImplicitDecoder(self.height, self.width, ngf=int(cfg["ngf_opac"]),
                                                in_ch=20, out_ch=c).to(self.device)
        self.opt = torch.optim.AdamW(self.decoder.parameters(), lr=cfg["depth_lr_base"])
        self.sched = torch.optim.lr_scheduler.CyclicLR(
            self.opt, base_lr=cfg["depth_lr_base"], max_lr=cfg["depth_lr_max"], mode="triangular2",
            step_size_up=int(cfg["depth_lr_step"]), cycle_momentum=False)
        self.opt_opac = torch.optim.AdamW(self.decoder_opac.parameters(), lr=cfg["opac_lr_base"])
        self.sched_opac = torch.optim.lr_scheduler.CyclicLR(
            self.opt_opac, base_lr=cfg["opac_lr_base"], max_lr=cfg["opac_lr_max"],
            mode="triangular2", step_size_up=int(cfg["opac_lr_step"]), cycle_momentum=False)

    def residual(self):
        return masked_channel_sum(self.decoder(self.inp), self.mask_norm, self.cfg["depth_scale"])

    def learned_opacity(self):
        return masked_channel_sum(self.decoder_opac(self.inp), self.mask_norm,
                                  self.cfg["opac_scale"])

    def gaussians(self):
        """Eq. 3 means and decoder opacity logits of all N*H*W Gaussians (with grad)."""
        means = self.xyz(self.z + self.residual())
        logits = self.opac0_logit + self.learned_opacity()
        return means, logits[:, 0]

    def step_decoders(self):
        """Official order: opacity decoder step/zero/sched, then depth decoder step/zero/sched."""
        for opt, sched in ((self.opt_opac, self.sched_opac), (self.opt, self.sched)):
            opt.step()
            opt.zero_grad()
            sched.step()

    def zero_decoder_grads(self):
        self.opt_opac.zero_grad()
        self.opt.zero_grad()

    # ------------------------------------------------------------ scales (Eq. 9)
    @torch.no_grad()
    def log_scales(self, use_decoder):
        """Official set_scaling: log(|tan(FoVy/2) * D / H| * radius_mult), isotropic."""
        depth = self.z + (self.residual() if use_decoder else 0.0)
        fovy = 2 * math.atan(self.height / (2 * float(self.K[1, 1])))
        radii = math.tan(0.5 * fovy) * depth / self.height
        return torch.log(torch.sqrt(radii ** 2) * self.cfg["radius_mult"]).repeat(1, 3)

    # ------------------------------------------------------------ flow loss (Eq. 6 / Eq. 8)
    def warping_loss(self, xyz, order=None):
        """Official warping_loss: shuffle views, idx1 = first, sum_{idx2} L1mean(M*warp, M*X0)."""
        if order is None:
            order = list(range(self.n))
            self.rng.shuffle(order)
        img = xyz.reshape(self.n, self.height, self.width, 3).permute(0, 3, 1, 2)
        i = order[0]
        x0 = img[i:i + 1]
        loss = 0
        for j in order[1:]:
            m = self.flow_masks[i, j][None]
            warped = backwarp(img[j:j + 1], self.flows[i, j][None])
            loss = loss + F.l1_loss(warped * m, x0 * m)
        return loss

    def align(self):
        """Coarse alignment (Sec. 4.3, Eq. 8): per-view scale/offset of the monocular depth."""
        cfg = self.cfg
        s = torch.ones(self.n, device=self.device, requires_grad=True)
        o = torch.zeros(self.n, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([{"params": [s], "lr": cfg["align_lr_scale"]},
                                {"params": [o], "lr": cfg["align_lr_offset"]}])
        steps = int(cfg["align_steps"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        history = []
        for _ in range(steps):
            z = self.z_mono * s.repeat_interleave(self.hw)[:, None] + o.repeat_interleave(self.hw)[:, None]
            loss = self.warping_loss(self.xyz(z))
            loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()
            history.append(float(loss.detach()))
        with torch.no_grad():
            self.align_scale, self.align_offset = s.detach().clone(), o.detach().clone()
            self.z = (self.z_mono * s.repeat_interleave(self.hw)[:, None]
                      + o.repeat_interleave(self.hw)[:, None]).contiguous()
        return history

    # ------------------------------------------------------------ schedules
    def official_iteration(self, step):
        return int(self.cfg["iteration_offset"]) + int(step)

    def sh_degree(self, step):
        """SH +1 at official iterations 15000/16000/17000 (train.py trn_it_lst[1] + 0/1000/2000)."""
        return sum(int(step >= s) for s in self.cfg["sh_steps"])

    def samples(self, step):
        """Multisampling factor: randint(1, 4) for official iteration > 12000, else 1."""
        if step > int(self.cfg["multisample_start"]):
            return self.rng.randint(1, int(self.cfg["multisample_max"]))
        return 1

    def state_dict(self):
        out = {"align_scale": self.align_scale.cpu(), "align_offset": self.align_offset.cpu(),
               "step": int(self.step or 0)}
        if self.decoder is not None:
            out["decoder"] = {k: v.cpu() for k, v in self.decoder.state_dict().items()}
            out["decoder_opac"] = {k: v.cpu() for k, v in self.decoder_opac.state_dict().items()}
        return out
