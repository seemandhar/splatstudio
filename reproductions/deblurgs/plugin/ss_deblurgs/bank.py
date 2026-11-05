"""Per-image camera-motion state of DeblurGS: Bezier trajectories in se(3) and sub-frame
alignment parameters nu, held in a process-local bank.

Port of scene/motion.py (CameraMotionModule, curve_type "se3"), scene/bezier.py (BezierModel)
and the curve parts of train.py / gaussian_model.update_learning_rate of
github.com/taekkii/deblurgs @e0587475. The density component creates the bank from the
training cameras (``set_training_cameras``), publishes it as ``ACTIVE`` for the renderer, and
steps its optimizer after every runtime step. ``ss-eval`` has no bank (``ACTIVE is None``), so
every camera is rendered sharp (paper Sec. 3.4: motion parameters are not used at inference).
"""

import math

import torch
from torch import nn

from .pytorch3d_se3 import se3_exp_map, se3_log_map

ACTIVE = None  # live bank of the current training process (None in ss-eval)

BANK_DEFAULTS = {
    "num_subframes": 21,           # paper Sec. 4.1, args num_subframes
    "curve_order": 9,              # paper Sec. 4.1, args curve_order
    "curve_rotation_lr": 1e-3,     # paper Sec. 4.1, args curve_rotation_lr
    "curve_controlpoints_lr": 1e-2,  # paper Sec. 4.1, args curve_controlpoints_lr
    "curve_alignment_lr": 3e-3,    # paper Sec. 4.1 (code default 0.0, comment "# 3e-3")
    "curve_alignment_start": 30000,  # args curve_alignment_start (get_scheduler step_warmup)
    "curve_alignment_final_lr": 1e-7,  # train.py alignment_func lr_final
    "curve_lr_half_iter": 15000,   # paper Sec. 4.1, args curve_lr_half_iter
    "curve_start_iter": 1000,      # args curve_start_iter
    "curve_end_iter": 100000,      # args curve_end_iter
    "curve_init_noise": 0.001,     # BezierModel initial_noise default
    "total_steps": 150000,         # args iterations (nu schedule step_final)
    "seed": 0,
}


def bezier_coefficients(t, order):
    """bezier.py _get_bezier_coeff: c_k(t) = binom(C,k) t^(C-k) (1-t)^k, k = 0..C."""
    k = torch.arange(order + 1, device=t.device, dtype=t.dtype)
    binom = torch.tensor([math.comb(order, int(i)) for i in range(order + 1)], device=t.device,
                         dtype=t.dtype)
    return (t[:, None] ** (order - k)) * ((1 - t)[:, None] ** k) * binom


def bezier(control_points, t):
    """control_points [C+1, d], t [f] -> [f, d] (BezierModel.forward)."""
    return (bezier_coefficients(t, control_points.shape[0] - 1)[:, :, None]
            * control_points[None]).sum(dim=1)


def viewmat_to_row_c2w(viewmat):
    """OpenCV w2c [n,4,4] -> pytorch3d row-vector c2w: M[:3,:3] = R_c2w^T, M[3,:3] = C."""
    c2w = torch.linalg.inv(viewmat)
    m = torch.zeros_like(viewmat)
    m[:, :3, :3] = c2w[:, :3, :3].transpose(-2, -1)
    m[:, 3, :3] = c2w[:, :3, 3]
    m[:, 3, 3] = 1.0
    return m


def row_c2w_to_viewmat(m):
    """motion._sample_c2w_from_nu + _c2w_to_minicam: R_c2w = M[:3,:3]^T, C = M[3,:3];
    OpenCV w2c = [R_c2w^T | -R_c2w^T C]."""
    r_w2c = m[:, :3, :3]  # = (R_c2w)^T
    center = m[:, 3, :3]
    view = torch.zeros_like(m)
    view[:, :3, :3] = r_w2c
    view[:, :3, 3] = -(r_w2c @ center[:, :, None])[:, :, 0]
    view[:, 3, 3] = 1.0
    return view


def expon(step, init, final, max_steps):
    """general_utils.get_expon_lr_func (DeblurGS 'changed behaviour', no delay)."""
    if step < 0:
        return init
    if step > max_steps:
        return final
    if init <= 0.0:
        return 0.0
    if init <= final:
        return init
    final = final if final > 0.0 else 1e-6
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(init) * (1 - t) + math.log(final) * t)


def alignment_lr(step, cfg):
    """train.py alignment_func = get_scheduler(lr_init, 1e-7, 0, step_warmup, step_final)."""
    lr0, start, final = (cfg["curve_alignment_lr"], cfg["curve_alignment_start"],
                         cfg["total_steps"])
    if step <= start:
        return 0.0
    if step <= final:
        if lr0 <= 1e-8:
            return 0.0
        rate = math.log(cfg["curve_alignment_final_lr"] / lr0) / (final - start)
        return lr0 * math.exp(rate * (step - start))
    return cfg["curve_alignment_final_lr"]


def curve_lr(step, lr0, cfg):
    """update_learning_rate: from curve_start_iter, lr *= 0.5^(1/half) once per iteration."""
    n = max(0, step - cfg["curve_start_iter"] + 1)
    return lr0 * 0.5 ** (n / cfg["curve_lr_half_iter"])


class TrajectoryBank(nn.Module):
    def __init__(self, cameras, cfg, device):
        super().__init__()
        self.cfg = {**BANK_DEFAULTS, **cfg}
        c = self.cfg
        f, order = int(c["num_subframes"]), int(c["curve_order"])
        if f < 3 or order < 1:
            raise ValueError("DeblurGS needs >= 3 sub-frames and a curve order >= 1")
        viewmats = torch.stack([cam["viewmat"].float() for cam in cameras]).to(device)
        self.register_buffer("viewmats", viewmats)
        self.width, self.height = int(cameras[0]["width"]), int(cameras[0]["height"])
        self.register_buffer("Ks", torch.stack([cam["K"].float() for cam in cameras]).to(device))
        log = se3_log_map(viewmat_to_row_c2w(viewmats))  # [n, 6] = [log_trans | log_rot]
        g = torch.Generator(device=device).manual_seed(int(c["seed"]))
        n = len(cameras)

        def init(points):  # BezierModel: repeat to C+1 control points + N(0, noise^2)
            rep = points[:, None, :].repeat(1, order + 1, 1)
            noise = torch.randn(rep.shape, generator=g, device=device) * c["curve_init_noise"]
            return nn.Parameter((rep + noise).contiguous())

        self.rot = init(log[:, 3:])
        self.trans = init(log[:, :3])
        lin = torch.linspace(1 / (f - 1), 1.0 - 1 / (f - 1), f - 2, device=device)
        self.nu = nn.Parameter(torch.logit(lin)[None].repeat(n, 1).contiguous())
        self.optimizer = torch.optim.Adam(
            [{"params": [self.rot], "lr": c["curve_rotation_lr"], "name": "curve_rot"},
             {"params": [self.trans], "lr": c["curve_controlpoints_lr"], "name": "curve_trans"},
             {"params": [self.nu], "lr": 0.0, "name": "curve_alignment"}], lr=0.0, eps=1e-15)
        self.step = 0  # last completed runtime step
        self.counts = {"curve_steps": 0}

    # -- queries -------------------------------------------------------------------------
    def view_index(self, viewmat):
        d = (self.viewmats - viewmat.to(self.viewmats.device).float()).abs().flatten(1).sum(1)
        i = int(torch.argmin(d))
        return i if float(d[i]) < 1e-4 else None

    def trainable(self, step):
        """alternate_optimization toggles at curve_start_iter and curve_end_iter."""
        return self.cfg["curve_start_iter"] <= step < self.cfg["curve_end_iter"]

    def times(self, idx):
        """_sample_nu_from_alignment: sort(clamp(cat[0, sigmoid(nu), 1], 0, 1)) (no jitter:
        curve_random_sample default False)."""
        nu = torch.sigmoid(self.nu[idx])
        z = torch.zeros(1, device=nu.device)
        return torch.cat([z, nu, z + 1]).clamp(0.0, 1.0).sort().values

    def subframe_times(self, idx, step):
        """train.py: subframe_indice 'all' from curve_start_iter, else 1 -> linspace(0, f-1, 1)
        .long() = [0], i.e. the first sorted time (0)."""
        t = self.times(idx)
        if step < self.cfg["curve_start_iter"]:
            t = t[:1]
        return t

    def poses(self, idx, t):
        """se3_exp_map(cat[trans(t), rot(t)]) -> OpenCV w2c viewmats [f, 4, 4]."""
        se3 = torch.cat([bezier(self.trans[idx], t), bezier(self.rot[idx], t)], dim=1)
        return row_c2w_to_viewmat(se3_exp_map(se3))

    def subframe_viewmats(self, idx, step):
        if self.trainable(step):
            return self.poses(idx, self.subframe_times(idx, step))
        with torch.no_grad():
            return self.poses(idx, self.subframe_times(idx, step))

    # -- optimisation --------------------------------------------------------------------
    def lrs(self, step):
        c = self.cfg
        return {"curve_rot": curve_lr(step, c["curve_rotation_lr"], c),
                "curve_trans": curve_lr(step, c["curve_controlpoints_lr"], c),
                "curve_alignment": alignment_lr(step, c)}

    @torch.no_grad()
    def after_step(self, step):
        """Adam step of the curve params of this iteration (official single optimizer: the
        params have grad None while frozen, so they are skipped)."""
        lrs = self.lrs(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lrs[group["name"]]
        if any(p.grad is not None for p in (self.rot, self.trans, self.nu)):
            self.optimizer.step()
            self.counts["curve_steps"] += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.step = step

    def export(self):
        return {"rot": self.rot.detach().cpu(), "trans": self.trans.detach().cpu(),
                "nu": self.nu.detach().cpu(), "viewmats": self.viewmats.detach().cpu(),
                "step": self.step, "curve_steps": self.counts["curve_steps"]}
