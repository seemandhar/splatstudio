"""CRiM-GS camera-motion kernel: per-image latent neural ODEs for the rigid (T_r) and adaptive
distortion-aware (T_a) transformations, plus the pixel-wise weight / scalar-mask CNN.

Paper: arXiv 2407.03923v2 Sec. 4.1-4.4 and appendix "Implementation Details". There is no
official CRiM-GS code; numerics the paper omits follow the authors' successor CoMoGaussian
(github.com/Jho-Yonsei/CoMoGaussian @cd6d536, scene/como_kernel.py, MIT), cited per item below.

The density component builds the kernel from the training cameras (``set_training_cameras``),
publishes it as ``ACTIVE`` for the renderer, and steps its optimizer after every runtime step.
``ss-eval`` has no kernel (``ACTIVE is None``): test views are rendered sharp at the input pose.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .ode import odeint_dopri5
from .se3 import homogeneous, normalize_axis, screw_exp

ACTIVE = None  # live kernel of the current training process (None in ss-eval)

KERNEL_DEFAULTS = {
    "num_poses": 9,             # App. A: N = 9
    "hidden": 64,               # App. A: embedding / encoder / decoder hidden size 64
    "exposure_time": 8.0,       # tau_i = linspace(0, exposure, N); CoMo integer times 0..N-1
    "rtol": 1e-4,               # CoMo DiffEqSolver odeint_rtol (paper: dopri5, tol not given)
    "atol": 1e-5,               # CoMo DiffEqSolver odeint_atol
    "max_ode_steps": 1000,
    "decoder_init": 1e-5,       # Sec. 4.3: U(-1e-5, 1e-5) (applied to all decoders, as CoMo)
    "cnn_channels": 64,         # App. A: 3 conv layers with 64 channels
    "cnn_kernels": (5, 3, 3),   # App. A: 5x5 then 3x3
    "lr": 1e-4,                 # CoMo CoMoKernel Adam lr (not in the paper)
    "lr_final": 1e-6,           # CoMo lr_factor = (1e-4 - 1e-6) / iterations
    "start_warp": 1000,         # App. A: transformations trained after 1k iterations
    "start_pixel_weight": 3000,  # App. A: pixel weight + mask after 3k iterations
    "total_steps": 40000,       # App. A: 40k iterations
    "seed": 0,
}


class LatentDerivative(nn.Module):
    """f / g of App. A: "two parallel single linear layers, one for the rotation and one for the
    translation", ReLU for non-linearity. CoMo NeuralDerivative order: ReLU(z) -> Linear per half.
    Autonomous (no explicit tau input; CoMo's per-step time embedding is a CoMo-only addition)."""

    def __init__(self, hidden):
        super().__init__()
        half = hidden // 2
        self.linear_rot = nn.Linear(half, half)
        self.linear_trans = nn.Linear(half, half)
        self.evaluations = 0

    def forward(self, t, z):
        self.evaluations += 1
        p, q = torch.chunk(torch.relu(z), 2, dim=-1)
        return torch.cat((self.linear_rot(p), self.linear_trans(q)), dim=-1)


class WeightCNN(nn.Module):
    """F of App. A (3 convs, 64 ch, 5x5/3x3/3x3; ReLU + InstanceNorm after each as in the
    BAGS/CoMo code), a pointwise weight head (softmax over the N renders) and a pointwise mask
    head averaged over the N axis (sigmoid)."""

    def __init__(self, channels=64, kernels=(5, 3, 3)):
        super().__init__()
        layers, cin = [], 3
        for k in kernels:
            layers += [nn.Conv2d(cin, channels, k, padding=k // 2), nn.ReLU(),
                       nn.InstanceNorm2d(channels)]
            cin = channels
        self.features = nn.Sequential(*layers)
        self.weight_head = nn.Conv2d(channels, 1, 1, bias=False)
        self.mask_head = nn.Conv2d(channels, 1, 1, bias=False)

    def forward(self, images):
        """images [N, H, W, 3] -> weights P [N, H, W, 1] (sum 1 over N), mask M [H, W, 1]."""
        x = images.permute(0, 3, 1, 2)
        feat = self.features(x)
        weights = F.softmax(self.weight_head(feat), dim=0)
        mask = torch.sigmoid(self.mask_head(feat).mean(dim=0, keepdim=True))
        return weights.permute(0, 2, 3, 1), mask[0].permute(1, 2, 0)


class CRiMKernel(nn.Module):
    def __init__(self, cameras, cfg, device):
        super().__init__()
        self.cfg = {**KERNEL_DEFAULTS, **cfg}
        c = self.cfg
        h = int(c["hidden"])
        if h % 2 or h < 4:
            raise ValueError("hidden must be even (rotation / translation halves)")
        if int(c["num_poses"]) < 2:
            raise ValueError("CRiM-GS needs at least 2 poses along the trajectory")
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(int(c["seed"]))
        viewmats = torch.stack([cam["viewmat"].float() for cam in cameras]).to(device)
        self.register_buffer("viewmats", viewmats)
        self.register_buffer("c2ws", torch.linalg.inv(viewmats))
        n = len(cameras)
        half = h // 2
        self.embedding = nn.Embedding(n, h)                  # App. A: nn.Embedding, size 64
        self.encoder_rigid = nn.Linear(h, h)                 # E_r (single layer)
        self.encoder_adaptive = nn.Linear(h, h)              # E_a (single layer)
        self.f = LatentDerivative(h)                         # rigid neural derivative
        self.g = LatentDerivative(h)                         # adaptive neural derivative
        self.dec_rigid_w = nn.Linear(half, 3)                # D_r: unit axis (normalised)
        self.dec_rigid_theta = nn.Linear(half, 1)            # D_r: angle (independent)
        self.dec_rigid_v = nn.Linear(half, 3)                # D_r: translation component
        self.dec_adaptive_A = nn.Linear(half, 9)             # D_a: A (R_a = A + I)
        self.dec_adaptive_t = nn.Linear(half, 3)             # D_a: t_a
        for layer in (self.dec_rigid_w, self.dec_rigid_theta, self.dec_rigid_v,
                      self.dec_adaptive_A, self.dec_adaptive_t):
            nn.init.uniform_(layer.weight, -c["decoder_init"], c["decoder_init"])
            nn.init.zeros_(layer.bias)
        self.cnn = WeightCNN(int(c["cnn_channels"]), tuple(c["cnn_kernels"]))
        torch.random.set_rng_state(gen_state)
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(c["lr"]))
        self.step = 0  # last completed runtime step
        self.counts = {"kernel_steps": 0, "ode_accepted": 0, "ode_rejected": 0}

    # -- schedule ----------------------------------------------------------------------
    def trajectory_active(self, step):
        return step > int(self.cfg["start_warp"])

    def pixel_weight_active(self, step):
        return step > int(self.cfg["start_pixel_weight"])

    def lr(self, step):
        """CoMo adjust_lr: lr -= (lr0 - lr_final) / iterations after every kernel step."""
        c = self.cfg
        k = max(0, step - int(c["start_warp"]) - 1)  # kernel steps already taken
        return float(c["lr"]) - k * (float(c["lr"]) - float(c["lr_final"])) / int(c["total_steps"])

    # -- queries -------------------------------------------------------------------------
    def view_index(self, viewmat):
        d = (self.viewmats - viewmat.to(self.viewmats.device).float()).abs().flatten(1).sum(1)
        i = int(torch.argmin(d))
        return i if float(d[i]) < 1e-4 else None

    def times(self):
        c = self.cfg
        return torch.linspace(0.0, float(c["exposure_time"]), int(c["num_poses"])).tolist()

    def latents(self, idx):
        """Eq. 5 / Sec. 4.3: z(tau_s) = z(tau_0) + int f, from z(tau_0) = E(embed(i))."""
        e = self.embedding.weight[idx]
        stats = {}
        z_r = odeint_dopri5(self.f, self.encoder_rigid(e), self.times(), self.cfg["rtol"],
                            self.cfg["atol"], int(self.cfg["max_ode_steps"]), stats)
        z_a = odeint_dopri5(self.g, self.encoder_adaptive(e), self.times(), self.cfg["rtol"],
                            self.cfg["atol"], int(self.cfg["max_ode_steps"]), stats)
        self.counts["ode_accepted"] += stats.get("accepted", 0)
        self.counts["ode_rejected"] += stats.get("rejected", 0)
        return z_r, z_a

    def decode(self, z_r, z_a):
        """D_r -> (w_hat, theta, v) -> T_r = e^{[S]theta} (Eq. 7-10);
        D_a -> (A, t_a), R_a = A + I -> T_a. Returns T_r, T_a [N,4,4] and R_a [N,3,3]."""
        zr_rot, zr_trans = torch.chunk(z_r, 2, dim=-1)
        w_hat = normalize_axis(self.dec_rigid_w(zr_rot))
        theta = self.dec_rigid_theta(zr_rot)
        v = self.dec_rigid_v(zr_trans)
        T_r = screw_exp(w_hat, theta, v)
        za_rot, za_trans = torch.chunk(z_a, 2, dim=-1)
        eye = torch.eye(3, device=z_a.device, dtype=z_a.dtype)
        R_a = self.dec_adaptive_A(za_rot).reshape(-1, 3, 3) + eye
        T_a = homogeneous(R_a, self.dec_adaptive_t(za_trans))
        return T_r, T_a, R_a

    def trajectory(self, idx):
        """Eq. 1 with T = camera-to-world: T^{tau_s} = T^{tau_0} T_r^{tau_s} T_a^{tau_s}.
        Returns OpenCV world-to-camera viewmats [N,4,4] and R_a [N,3,3]."""
        z_r, z_a = self.latents(idx)
        T_r, T_a, R_a = self.decode(z_r, z_a)
        c2w = self.c2ws[idx] @ T_r @ T_a
        return torch.linalg.inv(c2w), R_a

    def weights_and_mask(self, renders):
        return self.cnn(renders)

    # -- optimisation --------------------------------------------------------------------
    @torch.no_grad()
    def after_step(self, step):
        """CoMo train.py: kernel optimizer step + zero_grad + adjust_lr when step > start_warp."""
        if self.trajectory_active(step):
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr(step)
            if any(p.grad is not None for p in self.parameters()):
                self.optimizer.step()
                self.counts["kernel_steps"] += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.step = step

    def export(self):
        return {"state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                "step": self.step, **{k: int(v) for k, v in self.counts.items()}}
