"""ASG appearance field of Spec-Gaussian (real-scene configuration of the official code).

Ports of github.com/ingra14m/Specular-Gaussians @0b40ce3 (MIT, (c) 2024 Ziyi Yang):
utils/quaternion_utils.py:init_predefined_omega, utils/spec_utils.py:RenderingEquationEncoding,
positional_encoding, ASGRenderReal and SpecularNetworkReal (used by train.py with
``--is_real --asg_degree 12`` for every Mip-NeRF 360 scene, ``--is_indoor`` for the four indoor
scenes). Paper: Sec. 3.2, Eq. (ASG), (6), (8).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

DEFAULTS = {
    "feat_dim": 12,  # --asg_degree 12 (run_wo_anchor.sh); SpecularNetworkReal.asg_feature
    "num_theta": 2,  # SpecularNetworkReal: N = num_theta * num_phi = 8 ASGs
    "num_phi": 4,
    "hidden": 32,  # SpecularNetworkReal.hidden_feature
    "view_pe": 2,  # positional-encoding order of the view direction (paper Sec. 4.1)
    "indoor": False,  # --is_indoor: one more hidden layer in the decoder
}


# ------------------------------------------------------------------ fixed anisotropic basis
def _quat_product(p, q):
    pr, pi = p[..., :1], p[..., 1:]
    qr, qi = q[..., :1], q[..., 1:]
    out_r = pr * qr - (pi * qi).sum(dim=-1, keepdim=True)
    out_i = pr * qi + qr * pi + torch.linalg.cross(pi, qi, dim=-1)
    return torch.cat([out_r, out_i], dim=-1)


def _quat_rotate(p, q):
    q_inv = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return _quat_product(_quat_product(q, p), q_inv)


def _sph(theta, phi):
    return torch.stack([torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta),
                        torch.cos(theta)], dim=-1)


def init_predefined_omega(n_theta, n_phi):
    """Lobe z, tangent x (= omega_la) and bi-tangent y (= omega_mu) axes, each [n_theta*n_phi, 3].

    quaternion_utils.init_predefined_omega: theta in linspace(0, pi, n_theta), phi in
    linspace(0, 2 pi, n_phi); x = sph(theta + pi/2, phi); y = x rotated by pi/2 about z.
    """
    zs, xs, ys = [], [], []
    for theta in torch.linspace(0, math.pi, n_theta):
        for phi in torch.linspace(0, math.pi * 2, n_phi):
            z = _sph(theta, phi).view(1, 3)
            x = _sph(theta + math.pi / 2, phi).view(1, 3)
            p = torch.cat([torch.zeros(1, 1), x], dim=-1)
            half = torch.tensor(math.pi / 2) / 2
            q = torch.cat([torch.cos(half).view(1, 1), torch.sin(half) * z], dim=-1)
            y = _quat_rotate(p, q)[..., 1:]
            zs.append(z)
            xs.append(x)
            ys.append(y)
    return torch.cat(zs), torch.cat(xs), torch.cat(ys)


def asg_encoding(direction, a, la, mu, z, x, y):
    """kappa = (+)_i ASG(nu | [x,y,z], [lambda_i, mu_i], xi_i) (paper Eq. ASG / 8).

    direction [N,3] (unit); a [N,T,P,2] lobe amplitude xi; la, mu [N,T,P,1] raw sharpness;
    z/x/y [T,P,3] fixed axes. lambda = softplus(la - 1), mu = softplus(mu - 1) (official).
    Returns [N, T*P*2].
    """
    d = direction[:, None, None]
    smooth = F.relu((d * z).sum(-1, keepdim=True))  # S(nu; z) = max(nu.z, 0)
    lam = F.softplus(la - 1)
    mmu = F.softplus(mu - 1)
    expo = -lam * (x * d).sum(-1, keepdim=True).pow(2) - mmu * (y * d).sum(-1, keepdim=True).pow(2)
    return (a * smooth * torch.exp(expo)).reshape(direction.shape[0], -1)


def positional_encoding(v, freqs):
    """spec_utils.positional_encoding: [sin(v 2^k) ..., cos(v 2^k) ...], k < freqs."""
    bands = 2 ** torch.arange(freqs, device=v.device, dtype=v.dtype)
    pts = (v[..., None] * bands).reshape(*v.shape[:-1], freqs * v.shape[-1])
    return torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)


class SpecularNetworkReal(nn.Module):
    """Theta (Linear f -> {xi, lambda, mu}_N) + ASG encoding + decoder Psi -> c_s."""

    def __init__(self, feat_dim=12, num_theta=2, num_phi=4, hidden=32, view_pe=2, indoor=False):
        super().__init__()
        self.num_theta, self.num_phi, self.view_pe = num_theta, num_phi, view_pe
        self.gaussian_feature = nn.Linear(feat_dim, num_theta * num_phi * 4)  # Theta, Eq. 6
        in_ch = 2 * view_pe * 3 + 3 + num_theta * num_phi * 2
        layers = [nn.Linear(in_ch, hidden), nn.ReLU(inplace=True)]
        if indoor:
            layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden, 3)]
        self.mlp = nn.Sequential(*layers)  # Psi, Eq. 8
        nn.init.constant_(self.mlp[-1].bias, 0)
        z, x, y = init_predefined_omega(num_theta, num_phi)
        self.register_buffer("omega", z.view(num_theta, num_phi, 3), persistent=False)
        self.register_buffer("omega_la", x.view(num_theta, num_phi, 3), persistent=False)
        self.register_buffer("omega_mu", y.view(num_theta, num_phi, 3), persistent=False)

    def kappa(self, feat, viewdir):
        p = self.gaussian_feature(feat).view(-1, self.num_theta, self.num_phi, 4)
        a, la, mu = torch.split(p, [2, 1, 1], dim=-1)
        # ASGRenderReal queries the ASGs with the view direction d (camera -> Gaussian).
        return asg_encoding(viewdir, a, la, mu, self.omega, self.omega_la, self.omega_mu)

    def forward(self, feat, viewdir):
        """feat [N,F], viewdir [N,3] unit (camera -> Gaussian). Returns c_s [N,3]."""
        inp = [self.kappa(feat, viewdir), viewdir]
        if self.view_pe > 0:
            inp.append(positional_encoding(viewdir, self.view_pe))
        return self.mlp(torch.cat(inp, dim=-1))


def build_network(cfg):
    return SpecularNetworkReal(int(cfg["feat_dim"]), int(cfg["num_theta"]), int(cfg["num_phi"]),
                               int(cfg["hidden"]), int(cfg["view_pe"]), bool(cfg["indoor"]))


def linear_lr(lr_init, lr_final, max_steps):
    """utils/general_utils.get_linear_noise_func with lr_delay_steps=0 (delay_rate 1)."""

    def helper(step):
        t = min(max(step / max_steps, 0.0), 1.0)
        return lr_init * (1 - t) + lr_final * t

    return helper


# ------------------------------------------------------------------ decoder <-> state blob
MAGIC = 9090.0
HEADER = 8


def pack(network, step=0):
    """Flatten the decoder into a 1-D tensor: header (magic, length, checksum, last completed
    training step) + weights."""
    flat = torch.cat([t.detach().float().reshape(-1) for t in network.state_dict().values()])
    head = torch.zeros(HEADER, device=flat.device)
    head[0] = MAGIC
    head[1] = float(flat.numel())
    head[2] = flat.double().sum().float()
    head[3] = float(step)
    return torch.cat([head, flat])


def baked_step(blob):
    return int(blob.reshape(-1)[3])


def unpack(blob, cfg):
    flat = blob.reshape(-1)
    if flat.numel() < HEADER or float(flat[0]) != MAGIC:
        raise ValueError("spec_blob holds no Spec-Gaussian decoder (state was not baked)")
    n = int(flat[1])
    if HEADER + n > flat.numel():
        raise ValueError("spec_blob is shorter than its recorded decoder length")
    weights = flat[HEADER:HEADER + n]
    if not torch.isclose(weights.double().sum().float(), flat[2], rtol=1e-4, atol=1e-3):
        raise ValueError("spec_blob checksum mismatch (rows reordered or corrupted)")
    net = build_network(cfg).to(blob.device)
    state, i = {}, 0
    for k, t in net.state_dict().items():
        state[k] = weights[i:i + t.numel()].view_as(t).clone()
        i += t.numel()
    if i != n:
        raise ValueError("spec_blob length disagrees with the decoder configuration")
    net.load_state_dict(state)
    return net.eval()
