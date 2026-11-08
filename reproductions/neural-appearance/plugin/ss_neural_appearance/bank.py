"""Run-wide learned state the native runtime has no slot for (component-owned, like
reproductions/splatfacto-w): the shared residual MLP (C1) and the PPISP module (C3, vendored
nv-tlabs/ppisp) with their optimizers and schedules, plus the training-view -> PPISP frame map.

Lifecycle: created by the density in ``set_training_cameras`` and published as ``ACTIVE`` for the
renderer and the regularizer loss; stepped by the density in ``after_backward``; at the last step
packed into the per-Gaussian field ``na_blob`` (and ``density.state``) so that plain ``ss-eval``,
which only restores ``params``, rebuilds it (``unpack``).
"""

import torch

from . import appearance as A

ACTIVE = None
MAGIC = 924.0  # blob header tag
VERSION = 1.0

PPISP_DEFAULTS = {
    "use": True,  # MODEL.PPISP.USE
    "controller_training_steps": 5000,  # CONTROLLER_TRAINING_STEPS
    "controller_distillation": True,  # CONTROLLER_DISTILLATION
}


def _ppisp_config(train_steps, cfg):
    from .ppisp_vendor import PPISPConfig

    config = PPISPConfig(use_controller=cfg["controller_training_steps"] > 0,
                         controller_distillation=cfg["controller_distillation"])
    total = train_steps + (cfg["controller_training_steps"] if cfg["controller_distillation"] else 0)
    # Optim/ppisp.py PPISPWrapper.initialize
    config.controller_activation_ratio = max(
        (total - cfg["controller_training_steps"]) / total, 0.0)
    return config, total


class Bank:
    def __init__(self, mode, appearance_cfg, ppisp_cfg, cameras, train_steps, mlp_lr=0.003,
                 mlp_betas=(0.9, 0.99), seed=1337, device="cuda"):
        self.mode = mode  # "neural" | "sh"
        self.cfg = dict(appearance_cfg)
        self.ppisp_cfg = dict(ppisp_cfg)
        self.step = 0
        self.train_steps = int(train_steps)
        self.view_keys = {}
        for i, cam in enumerate(cameras or []):
            self.view_keys[self.key(cam["viewmat"])] = i
        self.mlp = None
        self.mlp_opt = None
        if mode == "neural":
            g = torch.Generator().manual_seed(seed)
            self.mlp = A.ResidualMLP(self.cfg, generator=g).to(device)
            # Neural.param_groups: lr 0.003, betas (0.9, 0.99), eps 1e-15 (FusedAdam)
            self.mlp_opt = torch.optim.Adam(self.mlp.parameters(), lr=mlp_lr, betas=mlp_betas,
                                            eps=1e-15)
        self.ppisp = None
        self.ppisp_opts, self.ppisp_scheds = [], []
        if self.ppisp_cfg["use"]:
            from .ppisp_vendor import PPISP

            config, total = _ppisp_config(self.train_steps, self.ppisp_cfg)
            # one physical camera for every scene used here (M360 / T&T / DB); frames = train views
            self.ppisp = PPISP(1, max(len(self.view_keys), 1), config)
            self.ppisp_opts = self.ppisp.create_optimizers()
            self.ppisp_scheds = self.ppisp.create_schedulers(self.ppisp_opts, total)

    @staticmethod
    def key(viewmat):
        return tuple(round(float(v), 5) for v in viewmat.detach().reshape(-1).cpu().tolist())

    def frame_index(self, viewmat):
        """PPISP frame of a training view, -1 (novel view: controller) otherwise."""
        return self.view_keys.get(self.key(viewmat), -1)

    def apply_ppisp(self, rgb_hwc, camera, novel=False):
        if self.ppisp is None:
            return rgb_hwc
        frame = -1 if novel else self.frame_index(camera["viewmat"])
        return self.ppisp(rgb_hwc.contiguous(), camera_idx=0, frame_idx=frame)

    def ppisp_regularization(self):
        return None if self.ppisp is None else self.ppisp.get_regularization_loss()

    def optimizer_step(self, step):
        """Neural MLP Adam (only through train_steps) + PPISP optimizers/scheduler (every step,
        PPISPWrapper.step)."""
        if self.mlp_opt is not None and step <= self.train_steps:
            self.mlp_opt.step()
        if self.mlp_opt is not None:
            self.mlp_opt.zero_grad(set_to_none=True)
        for opt in self.ppisp_opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in self.ppisp_scheds:
            sched.step()
        self.step = step

    def set_mlp_lr(self, lr):
        if self.mlp_opt is not None:
            self.mlp_opt.param_groups[0]["lr"] = float(lr)

    def controller_trained(self):
        """PPISP controller has been distilled (step >= activation step)."""
        if self.ppisp is None:
            return False
        p = self.ppisp
        return (p.config.use_controller and p.config.controller_activation_ratio < 1.0
                and p._controller_activation_step >= 0
                and (p._ppisp_scheduler is None
                     or p._ppisp_scheduler.last_epoch >= p._controller_activation_step + 1))

    def distilling(self):
        """True during the PPISP controller distillation steps (after train_steps)."""
        return self.ppisp is not None and self.step >= self.train_steps

    # ---------------------------------------------------------------- persistence
    def tensors(self):
        out = {}
        if self.mlp is not None:
            for i, w in enumerate(self.mlp.weights):
                out[f"mlp.{i}"] = w.detach()
        if self.ppisp is not None:
            for k, v in self.ppisp.state_dict().items():
                out[f"ppisp.{k}"] = v.detach()
        return out

    def pack(self):
        """Flat float32 vector: [MAGIC, VERSION, mode, ppisp?, n_frames, F, k, neurons, layers,
        controller_trained, n_deg, degrees..., payload (sorted keys)]."""
        degrees = list(self.cfg["sh_degrees"])
        header = [MAGIC, VERSION, 1.0 if self.mode == "neural" else 0.0,
                  1.0 if self.ppisp is not None else 0.0,
                  float(self.ppisp.num_frames if self.ppisp is not None else 0),
                  float(self.cfg["feature_dim"]), float(self.cfg["feature_frequencies"]),
                  float(self.cfg["n_neurons"]), float(self.cfg["n_hidden_layers"]),
                  1.0 if self.controller_trained() else 0.0,
                  float(len(degrees))] + [float(d) for d in degrees]
        tensors = self.tensors()
        device = next(iter(tensors.values())).device if tensors else "cpu"
        parts = [torch.tensor(header, dtype=torch.float32, device=device)]
        parts += [tensors[k].float().reshape(-1) for k in sorted(tensors)]
        return torch.cat(parts)


def unpack(flat, device="cuda"):
    """Rebuild an inference Bank from ``pack()`` output (prefix of the flattened na_blob)."""
    h = flat[:11].tolist()
    if abs(h[0] - MAGIC) > 1e-3 or abs(h[1] - VERSION) > 1e-3:
        raise ValueError("na_blob does not contain a packed neural-appearance state "
                         "(was the run trained to its final step?)")
    n_deg = int(h[10])
    degrees = [int(round(v)) for v in flat[11:11 + n_deg].tolist()]
    cfg = {"feature_dim": int(h[5]), "feature_frequencies": int(h[6]), "n_neurons": int(h[7]),
           "n_hidden_layers": int(h[8]), "sh_degrees": degrees}
    mode = "neural" if h[2] > 0.5 else "sh"
    n_frames = int(h[4])
    ppisp_cfg = dict(PPISP_DEFAULTS, use=h[3] > 0.5)
    cameras = [{"viewmat": torch.full((4, 4), float(i))} for i in range(n_frames)]
    bank = Bank(mode, cfg, ppisp_cfg, cameras, train_steps=30000, device=device)
    bank.view_keys = {}  # inference: every view is novel
    offset = 11 + n_deg
    tensors = bank.tensors()
    loaded = {}
    for k in sorted(tensors):
        n = tensors[k].numel()
        loaded[k] = flat[offset:offset + n].reshape(tensors[k].shape).to(device)
        offset += n
    with torch.no_grad():
        if bank.mlp is not None:
            for i, w in enumerate(bank.mlp.weights):
                w.copy_(loaded[f"mlp.{i}"])
        if bank.ppisp is not None:
            state = {k[len("ppisp."):]: v for k, v in loaded.items() if k.startswith("ppisp.")}
            bank.ppisp.load_state_dict(state)
            bank.ppisp._ppisp_scheduler = None  # inference mode: controller ready
            if h[9] < 0.5:  # run stopped before distillation: identity corrections (PPISP default)
                bank.ppisp.config.controller_activation_ratio = 1.0
            bank.ppisp_opts, bank.ppisp_scheds = [], []
    bank.step = bank.train_steps
    return bank
