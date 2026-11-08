"""SSS sampler and component recycling, ported from the official code:
`utils/sghmc.py` (AdamSGHMC, 'xyz' group), `train.py` (lr setup, densify block),
`scene/nt_model.py` (create_from_pcd, recycle_components, add_components, _update_params,
_sample_alives), `cuda_rasterizer/utils.cu compute_relocation_student_t`.

Means (paper Eq. sigmoidUpdate, as coded): the runtime's Adam for `means` runs with lr 0 so it only
keeps the Adam moments m, v; here
  g      = sqrt(1-b2^t)/(1-b1^t) m / (sqrt(v) + eps)          (Adam-preconditioned gradient)
  sig    = sigmoid(100 ((1 - |tanh o|) - 0.995))               (from the pre-step parameters)
  t <= B:  r = r sig - lr g - r sig + Sigma (N(0, sqrt(2 lr C_b)) sig)   (friction off, cov noise)
  t == B+1: r := 0;   t > B:  r = r sig - lr g - lr C r sig + N(0, sqrt(2 lr C)) sig
  mu    += lr r,  lr = exp-decay sqrt(lr_hi s/7.4518) -> sqrt(lr_lo s/7.4518) over 30k steps.
The momentum r lives in the means Adam state ('momentum' key) so population ops keep it aligned.

Recycling every 100 steps for 500 < step < 25000 (Sec. 3.4.2, Eq. relocation):
  dead = |o| < 0.005 (at most 5 % of N: the lowest |o|), sources ~ multinomial(|o| of alive);
  o_new = 1 - (1 - o)^(1/N), scale_new = (o term1 / K)^2 scale (code; same nu), copied to the dead
  and the sources; Adam moments of the sources zeroed, SGHMC momentum of the dead zeroed.
  Then add min(cap, int(1.05 N)) - N copies of multinomial(|o|) samples with the same relocation
  (new rows: zero moments and momentum; sources: zero Adam moments).
"""

import math

import torch

from .renderer import nu_activation, signed_opacity

OPTIONS = {
    "cap_max": 3_000_000,
    "C_burnin": 5e5,
    "C": 1.2e2,
    "burnin_iterations": 7000,
    "densify_from": 500,
    "densify_until": 25_000,
    "interval": 100,
    "dead_opacity": 0.005,
    "recycle_fraction": 0.05,
    "growth": 1.05,
    "n_max": 51,
    "sig_k": 100.0,
    "sig_x0": 0.995,
    "lr_init_base": 0.001226442339330813,
    "lr_final_base": 1.226630619638022e-05,
    "lr_reference_scale": 7.45176315307617,
    "lr_max_steps": 30_000,
    "betas": (0.9, 0.999),
    "adam_eps": 1e-15,
    "opacity_lr": 0.005,
    "nu_lr": 10.0,
    "nu_lr_step": 5000,
    "nu_lr_gamma": 0.5,
    "nu_lr_min": 0.01,
    "init_opacity": 0.5,
    "init_scale_factor": 0.1,
}
EPS = torch.finfo(torch.float32).eps


# ------------------------------------------------------------------------------- schedules
def expon_lr(step, lr_init, lr_final, max_steps):
    """get_expon_lr_func with lr_delay_steps = 0 (delay_mult unused)."""
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def sghmc_lr(step, scene_scale, cfg):
    """train.py: position_lr_{init,final} = sqrt(base s / s_ref) / s, scheduled lr = that * s."""
    s, ref = scene_scale, cfg["lr_reference_scale"]
    hi = math.sqrt(cfg["lr_init_base"] * s / ref)
    lo = math.sqrt(cfg["lr_final_base"] * s / ref)
    return expon_lr(step, hi, lo, cfg["lr_max_steps"])


def nu_lr(step, cfg):
    """get_step_lr_func(lr_init=10, step_size=5000, gamma=0.5, min_lr=0.01)."""
    lr = cfg["nu_lr"] * cfg["nu_lr_gamma"] ** int(step / cfg["nu_lr_step"])
    return max(cfg["nu_lr_min"], lr)


def op_sigmoid(x, k=100.0, x0=0.995):
    return 1 / (1 + torch.exp(-k * (x - x0)))


def inverse_tanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def covariance(params):
    from .projection import covariance3d

    return covariance3d(params["quats"], params["scales"])


# ------------------------------------------------------------------------------- SGHMC
@torch.no_grad()
def sghmc_update(means, momentum, exp_avg, exp_avg_sq, adam_step, iteration, lr, sig, cov, cfg,
                 generator=None):
    """One official AdamSGHMC 'xyz' update in place. Returns the new momentum (in place too)."""
    b1, b2 = cfg["betas"]
    bc1, bc2 = 1 - b1 ** adam_step, 1 - b2 ** adam_step
    step_size = math.sqrt(bc2) / bc1
    adam_gradient = -step_size * exp_avg / (exp_avg_sq.sqrt() + cfg["adam_eps"])
    gradient = -adam_gradient
    burnin = int(cfg["burnin_iterations"])
    if iteration == burnin + 1:
        momentum.zero_()
    if iteration <= burnin:
        sigma = math.sqrt(2 * lr * cfg["C_burnin"])
        sample = torch.randn(gradient.shape, device=gradient.device, generator=generator) * sigma
        noise = torch.bmm(cov, (sample * sig)[..., None])[..., 0]
        friction = lr * (1 / lr) * momentum * sig
    else:
        sigma = math.sqrt(2 * lr * cfg["C"])
        sample = torch.randn(gradient.shape, device=gradient.device, generator=generator) * sigma
        noise = sample * sig
        friction = lr * cfg["C"] * momentum * sig
    momentum.mul_(sig)
    momentum.add_(-lr * gradient - friction + noise)
    means.add_(lr * momentum)
    return momentum


# ------------------------------------------------------------------------------- relocation
def binomials(n_max, device):
    b = torch.zeros(n_max, n_max, device=device)
    for n in range(n_max):
        for k in range(n + 1):
            b[n, k] = math.comb(n, k)
    return b


def relocation_student_t(opacity_old, scale_old, nu, N, binoms):
    """compute_relocation_student_t (utils.cu), float32 like the kernel; N already clamped.
    K = sum_{i=1..N} sum_{k<i} C(i-1,k) (-1)^k o_new^(k+1) B(1/2, ((k+1)(nu+3)-1)/2), summed over
    i first through the cumulative binomial table (same terms as the kernel's double loop)."""
    N = N.long()
    o_new = 1.0 - torch.pow(1.0 - opacity_old, 1.0 / N.float())
    lg_half = math.lgamma(0.5)
    term1 = torch.exp(lg_half + torch.lgamma((nu + 2) / 2) - torch.lgamma((nu + 3) / 2))
    cum = torch.cumsum(binoms, dim=0)  # cum[m, k] = sum_{i-1 <= m} C(i-1, k)
    denom = torch.zeros_like(opacity_old)
    n_max = int(N.max().item()) if N.numel() else 1
    for k in range(n_max):
        weight = cum[N - 1, k]  # zero when k > N-1
        term2 = (-1.0) ** k * torch.pow(o_new, k + 1)
        term3 = torch.exp(lg_half + torch.lgamma(((k + 1) * (nu + 3) - 1) / 2)
                          - torch.lgamma((k + 1) * (nu + 3) / 2))
        denom = denom + weight * term2 * term3
    coeff = opacity_old * term1 / denom
    return o_new, (coeff ** 2)[:, None] * scale_old


def sample_alives(probs, num, alive_indices=None, generator=None):
    """_sample_alives: |p| normalised multinomial with replacement + bincount ratio."""
    probs = probs.abs() / (probs.abs().sum() + EPS)
    sampled = torch.multinomial(probs, num, replacement=True, generator=generator)
    if alive_indices is not None:
        sampled = alive_indices[sampled]
    ratio = torch.bincount(sampled)[:, None]
    return sampled, ratio


def updated_params(params, idxs, ratio, binoms, cfg):
    """_update_params: relocated (raw opacity, log scale) for sources idxs."""
    nu = nu_activation(params["degrees"][idxs])
    N = (ratio[idxs, 0] + 1).clamp(1, cfg["n_max"])
    o_new, s_new = relocation_student_t(signed_opacity(params["opacities"][idxs]),
                                        torch.exp(params["scales"][idxs]), nu, N, binoms)
    o_new = torch.clamp(o_new, max=1.0 - EPS, min=-1.0 + EPS)
    d = cfg["dead_opacity"]
    o_new = torch.where((o_new >= 0) & (o_new < d), torch.full_like(o_new, d), o_new)
    o_new = torch.where((o_new < 0) & (o_new > -d), torch.full_like(o_new, -d), o_new)
    return inverse_tanh(o_new), torch.log(s_new)


def _adam_states(optimizers, params):
    for name, p in params.items():
        yield name, p, optimizers[name].state.get(p)


@torch.no_grad()
def recycle(params, optimizers, binoms, cfg, generator=None):
    o = signed_opacity(params["opacities"])
    d = cfg["dead_opacity"]
    dead = (o < d) & (o > -d)
    if int(dead.sum()) == 0:
        return 0
    alive_idx = (~dead).nonzero(as_tuple=True)[0]
    dead_idx = dead.nonzero(as_tuple=True)[0]
    n = len(o)
    cap = int(cfg["recycle_fraction"] * n)
    if int(dead.sum()) > cap:
        dead_idx = torch.sort(o.abs())[1][:cap]
    if len(alive_idx) == 0 or len(dead_idx) == 0:
        return 0
    reinit, ratio = sample_alives(o[alive_idx], len(dead_idx), alive_idx, generator)
    new_raw_o, new_log_s = updated_params(params, reinit, ratio, binoms, cfg)
    for name, p in params.items():
        if name == "opacities":
            p.data[dead_idx] = new_raw_o
        elif name == "scales":
            p.data[dead_idx] = new_log_s
        else:
            p.data[dead_idx] = p.data[reinit]
    params["opacities"].data[reinit] = params["opacities"].data[dead_idx]
    params["scales"].data[reinit] = params["scales"].data[dead_idx]
    for name, p, st in _adam_states(optimizers, params):
        if not st:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                st[key][reinit] = 0
        if "momentum" in st:
            st["momentum"][dead_idx] = 0
    return len(dead_idx)


def _replace_param(params, optimizers, name, new_data, state_fn):
    old = params[name]
    new = torch.nn.Parameter(new_data.detach().clone().contiguous(),
                             requires_grad=old.requires_grad)
    opt = optimizers[name]
    st = opt.state.pop(old, None)
    if st is not None:
        for key in list(st.keys()):
            if key != "step":
                st[key] = state_fn(key, st[key])
        opt.state[new] = st
    for group in opt.param_groups:
        group["params"] = [new]
    params[name] = new


@torch.no_grad()
def add_components(params, optimizers, binoms, cfg, generator=None):
    n = len(params["means"])
    target = min(int(cfg["cap_max"]), int(cfg["growth"] * n))
    num = max(0, target - n)
    if num <= 0:
        return 0
    o = signed_opacity(params["opacities"])
    add_idx, ratio = sample_alives(o, num, None, generator)
    new_raw_o, new_log_s = updated_params(params, add_idx, ratio, binoms, cfg)
    params["opacities"].data[add_idx] = new_raw_o
    params["scales"].data[add_idx] = new_log_s
    for name in list(params.keys()):
        p = params[name]
        new_rows = p.data[add_idx]

        def state_fn(key, v, rows=num):
            if key in ("exp_avg", "exp_avg_sq"):
                v = v.clone()
                v[add_idx] = 0  # replace_tensors_to_optimizer(inds=add_idx)
            return torch.cat((v, torch.zeros((rows, *v.shape[1:]), device=v.device,
                                             dtype=v.dtype)))

        _replace_param(params, optimizers, name, torch.cat((p.data, new_rows)), state_fn)
    return num


def check_trainable(params):
    for name, p in params.items():
        if not (isinstance(p, torch.nn.Parameter) and p.is_leaf and p.requires_grad):
            raise RuntimeError(f"{name} is no longer a trainable leaf after density control")


# ------------------------------------------------------------------------------- component
class SGHMCRecycling:
    def __init__(self, spec):
        self.requires = set()
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown SSS density options: {sorted(unknown)}")
        cfg["betas"] = tuple(cfg["betas"])
        self.cfg = cfg
        self.state = {"recycled": 0, "added": 0, "events": 0, "scene_scale": 1.0}
        self.binoms = None
        self._sig = None
        self._cov = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        """create_from_pcd / training_setup: o = tanh^-1(0.5), scales log(0.1 sqrt(d2_3nn)),
        means handled by SGHMC (Adam lr 0 keeps only the moments), opacity lr 0.005."""
        c = self.cfg
        with torch.no_grad():
            params["opacities"].fill_(0.5 * math.log((1 + c["init_opacity"]) /
                                                     (1 - c["init_opacity"])))
            params["scales"].add_(math.log(c["init_scale_factor"]))
        for opt in optimizers.values():
            for g in opt.param_groups:
                g["eps"] = c["adam_eps"]
                g["betas"] = c["betas"]
        optimizers["means"].param_groups[0]["lr"] = 0.0
        optimizers["opacities"].param_groups[0]["lr"] = c["opacity_lr"]
        optimizers["degrees"].param_groups[0]["lr"] = nu_lr(1, c)
        self.state["scene_scale"] = float(scene_scale)
        self.binoms = binomials(int(c["n_max"]), params["means"].device)

    def before_backward(self, params, optimizers, step, outputs):
        """sig and covariance from the parameters before this step's optimiser update."""
        with torch.no_grad():
            o = signed_opacity(params["opacities"])[:, None]
            self._sig = op_sigmoid(1 - o.abs(), self.cfg["sig_k"], self.cfg["sig_x0"])
            self._cov = covariance(params)

    def sghmc_step(self, params, optimizers, step):
        p = params["means"]
        st = optimizers["means"].state.get(p)
        if not st or "exp_avg" not in st:
            raise RuntimeError("means Adam moments are missing; SGHMC needs them")
        if optimizers["means"].param_groups[0]["lr"] != 0.0:
            raise RuntimeError("means must not be updated by Adam (SGHMC replaces it)")
        if "momentum" not in st:
            st["momentum"] = torch.zeros_like(p)
        lr = sghmc_lr(step, self.state["scene_scale"], self.cfg)
        sghmc_update(p.data, st["momentum"], st["exp_avg"], st["exp_avg_sq"],
                     float(st["step"]), step, lr, self._sig, self._cov, self.cfg)
        self._sig = self._cov = None
        return lr

    def active(self, step):
        c = self.cfg
        return c["densify_from"] < step < c["densify_until"] and step % c["interval"] == 0

    def after_backward(self, params, optimizers, step, outputs):
        self.state["lr_means"] = self.sghmc_step(params, optimizers, step)
        optimizers["degrees"].param_groups[0]["lr"] = nu_lr(step + 1, self.cfg)
        if self.active(step):
            self.state["recycled"] += recycle(params, optimizers, self.binoms, self.cfg)
            self.state["added"] += add_components(params, optimizers, self.binoms, self.cfg)
            self.state["events"] += 1
            check_trainable(params)
            if step % 1000 == 0:
                torch.cuda.empty_cache()


def create_density(spec):
    return SGHMCRecycling(spec)
