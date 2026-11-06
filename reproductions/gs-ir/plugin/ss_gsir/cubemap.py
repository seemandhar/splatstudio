"""Cube-map texture operations used by GS-IR, re-implemented in PyTorch (+ one CUDA filter).

The official code uses nvdiffrast ``dr.texture(..., boundary_mode="cube")`` and nvdiffrec's
``diffuse_cubemap`` / ``specular_cubemap``. Layout: tensor [6, N, N, C] with faces
+X, -X, +Y, -Y, +Z, -Z (OpenGL convention, nvdiffrec ``cube_to_dir``), texel (x, y) at row y,
column x, texel centres at (i + 0.5) / N.

* ``sample``: bilinear ("linear") or nearest lookup by direction. Seamless filtering across
  face edges uses faces padded by one texel taken from the adjacent faces (nearest texel of the
  direction just beyond the edge); corner texels are approximated the same way.
* ``sample_mip``: "linear-mipmap-linear" with an explicit per-pixel mip level (nvdiffrast with
  ``mip_level_bias`` and no uv derivatives: level = bias, clamped to [0, levels-1]).
* ``mip_down``: 2x2 average with GS-IR's custom backward (pbr/light.py ``cubemap_mip``).
* ``diffuse_convolve``: nvdiffrec diffuse kernel sum_L L clamp(N.L, 0, 0.999) dOmega / 3.141592
  as a dense operator. ``specular_filter``: CUDA kernel (csrc), normalised by sum of weights.
"""

import numpy as np
import torch

from .cuda import ext


class _GatherRows(torch.autograd.Function):
    """src[idx] along dim 0 with an atomic index_add_ backward (advanced-indexing backward
    sorts the indices and is ~10x slower for the many duplicate texel lookups here)."""

    @staticmethod
    def forward(ctx, src, idx):
        ctx.shape = src.shape
        ctx.save_for_backward(idx)
        return src[idx]

    @staticmethod
    def backward(ctx, grad):
        (idx,) = ctx.saved_tensors
        out = torch.zeros(ctx.shape, device=grad.device, dtype=grad.dtype)
        out.index_add_(0, idx.reshape(-1), grad.reshape(-1, *ctx.shape[1:]))
        return out, None


def gather_rows(src, idx):
    if not src.requires_grad:
        return src[idx]
    return _GatherRows.apply(src, idx)


def face_dirs(face, x, y, n):
    """Unnormalised direction of (possibly fractional / out-of-range) texel coords on a face."""
    fx = 2.0 * ((x + 0.5) / n) - 1.0
    fy = 2.0 * ((y + 0.5) / n) - 1.0
    one = torch.ones_like(fx)
    table = {
        0: (one, -fy, -fx),
        1: (-one, -fy, fx),
        2: (fx, one, fy),
        3: (fx, -one, -fy),
        4: (fx, -fy, one),
        5: (-fx, -fy, -one),
    }
    return torch.stack(table[int(face)], dim=-1)


def texel_dirs(n, device, normalize=True):
    """[6, n, n, 3] directions of all texel centres."""
    ys, xs = torch.meshgrid(
        torch.arange(n, device=device, dtype=torch.float32),
        torch.arange(n, device=device, dtype=torch.float32),
        indexing="ij",
    )
    d = torch.stack([face_dirs(f, xs, ys, n) for f in range(6)])
    return torch.nn.functional.normalize(d, dim=-1) if normalize else d


def dir_to_face(d):
    """Direction [..., 3] -> (face long [...], s [...], t [...]) texture coords in [0, 1]."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    ax, ay, az = x.abs(), y.abs(), z.abs()
    is_x = (ax >= ay) & (ax >= az)
    is_y = ~is_x & (ay >= az)
    face = torch.where(
        is_x, torch.where(x >= 0, 0, 1), torch.where(is_y, torch.where(y >= 0, 2, 3),
                                                     torch.where(z >= 0, 4, 5))
    )
    sc = torch.where(is_x, torch.where(x >= 0, -z, z), torch.where(is_y, x, torch.where(z >= 0, x, -x)))
    tc = torch.where(is_x, -y, torch.where(is_y, torch.where(y >= 0, z, -z), -y))
    ma = torch.where(is_x, ax, torch.where(is_y, ay, az)).clamp_min(1e-12)
    return face, 0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0)


_PAD = {}


def pad_index(n, device):
    """[6, n+2, n+2] flat texel index (into [6*n*n]) of every padded texel."""
    key = (n, str(device))
    if key not in _PAD:
        r = torch.arange(-1, n + 1, device=device, dtype=torch.float32)
        ys, xs = torch.meshgrid(r, r, indexing="ij")
        out = []
        for f in range(6):
            face, s, t = dir_to_face(face_dirs(f, xs, ys, n))
            tx = (s * n).floor().long().clamp(0, n - 1)
            ty = (t * n).floor().long().clamp(0, n - 1)
            idx = (face * n + ty) * n + tx
            inner = (xs >= 0) & (xs < n) & (ys >= 0) & (ys < n)
            own = (f * n + ys.long().clamp(0, n - 1)) * n + xs.long().clamp(0, n - 1)
            out.append(torch.where(inner, own, idx))
        _PAD[key] = torch.stack(out)
    return _PAD[key]


def padded(tex):
    n, c = tex.shape[1], tex.shape[-1]
    return gather_rows(tex.reshape(-1, c), pad_index(n, tex.device))  # [6, n+2, n+2, C]


def _bilinear(pad, face, s, t):
    n = pad.shape[1] - 2
    c = pad.shape[-1]
    u = s * n - 0.5 + 1.0
    v = t * n - 0.5 + 1.0
    u0f, v0f = u.floor(), v.floor()
    fu, fv = (u - u0f)[..., None], (v - v0f)[..., None]
    u0 = u0f.long().clamp(0, n + 1)
    v0 = v0f.long().clamp(0, n + 1)
    u1 = (u0 + 1).clamp(max=n + 1)
    v1 = (v0 + 1).clamp(max=n + 1)
    flat = pad.reshape(-1, c)
    base = face * (n + 2)

    def g(vv, uu):
        return gather_rows(flat, (base + vv) * (n + 2) + uu)

    return (g(v0, u0) * (1 - fu) * (1 - fv) + g(v0, u1) * fu * (1 - fv)
            + g(v1, u0) * (1 - fu) * fv + g(v1, u1) * fu * fv)


def sample(tex, dirs, mode="linear"):
    """Look up cube texture [6, n, n, C] at directions [..., 3] -> [..., C]."""
    face, s, t = dir_to_face(dirs)
    n = tex.shape[1]
    if mode == "nearest":
        tx = (s * n).floor().long().clamp(0, n - 1)
        ty = (t * n).floor().long().clamp(0, n - 1)
        return gather_rows(tex.reshape(-1, tex.shape[-1]), (face * n + ty) * n + tx)
    return _bilinear(padded(tex), face, s, t)


def sample_mip(levels, dirs, level):
    """Trilinear lookup across a mip chain (list of [6, n_l, n_l, C]); level [...] float."""
    face, s, t = dir_to_face(dirs)
    lv = level.clamp(0.0, float(len(levels) - 1))
    l0 = lv.floor().long()
    l1 = (l0 + 1).clamp(max=len(levels) - 1)
    f = (lv - l0.float())[..., None]
    out = None
    for i, tex in enumerate(levels):
        w = (l0 == i).float()[..., None] * (1 - f) + (l1 == i).float()[..., None] * f
        if not bool((w > 0).any()):
            continue
        val = _bilinear(padded(tex), face, s, t) * w
        out = val if out is None else out + val
    return out


class _MipDown(torch.autograd.Function):
    """2x2 average pooling; backward = bilinear cube lookup of the coarse gradient x 0.25 at
    the fine texel directions (GS-IR pbr/light.py cubemap_mip.backward)."""

    @staticmethod
    def forward(ctx, cube):
        y = torch.nn.functional.avg_pool2d(cube.permute(0, 3, 1, 2), (2, 2))
        return y.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        dirs = texel_dirs(res, dout.device)
        return sample(dout * 0.25, dirs, "linear")


def mip_down(cube):
    return _MipDown.apply(cube)


def texel_area(n, device):
    """Per-texel solid-angle weight (nvdiffrec pixel_area) [n, n] (rows y, cols x)."""
    i = torch.arange(n, device=device, dtype=torch.float64)
    h = n // 2
    a = (i - h).abs()
    d = torch.atan((a + 1) / h) - torch.atan(a / h)
    return (d[None, :] * d[:, None]).float()


_DIFF = {}


def diffuse_operator(n, device):
    key = (n, str(device))
    if key not in _DIFF:
        d = texel_dirs(n, device).reshape(-1, 3)
        area = texel_area(n, device)[None].expand(6, n, n).reshape(-1)
        cos = (d @ d.T).clamp(0.0, 0.999)
        _DIFF[key] = cos * area[None, :] / 3.141592
    return _DIFF[key]


def diffuse_convolve(cube):
    n, c = cube.shape[1], cube.shape[-1]
    return (diffuse_operator(n, cube.device) @ cube.reshape(-1, c)).reshape(6, n, n, c)


_BOUNDS = {}


def ndf_cutoff(roughness, cutoff):
    """nvdiffrec __ndfBounds: cos(theta) keeping `cutoff` of the GGX(alpha^2 = r^4) energy."""
    costheta = np.cos(np.linspace(0, np.pi / 2.0, 1000000))
    a2 = roughness**4
    c = np.clip(costheta, 0.0, 1.0)
    d = (c * a2 - c) * c + 1.0
    D = np.cumsum(a2 / (d * d * np.pi))
    idx = np.argmax(D >= D[..., -1] * cutoff)
    return float(costheta[idx])


def spec_bounds(n, roughness, cutoff, device):
    key = (n, float(roughness), float(cutoff), str(device))
    if key not in _BOUNDS:
        cos_cut = ndf_cutoff(float(roughness), float(cutoff))
        with torch.cuda.device(device):
            _BOUNDS[key] = (cos_cut, ext().specular_bounds(n, cos_cut))
    return _BOUNDS[key]


class _Specular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cube, roughness, cos_cut, bounds):
        out = ext().specular_fwd(cube.contiguous(), bounds, float(roughness), float(cos_cut))
        ctx.save_for_backward(bounds)
        ctx.roughness, ctx.cos_cut = float(roughness), float(cos_cut)
        return out

    @staticmethod
    def backward(ctx, dout):
        (bounds,) = ctx.saved_tensors
        d = ext().specular_bwd(bounds, dout[..., :3].contiguous(), ctx.roughness, ctx.cos_cut)
        return d, None, None, None


def specular_filter(cube, roughness, cutoff=0.99):
    cos_cut, bounds = spec_bounds(cube.shape[1], roughness, cutoff, cube.device)
    out = _Specular.apply(cube, roughness, cos_cut, bounds)
    return out[..., 0:3] / out[..., 3:]


def latlong_dirs(h, w, device, centered=True):
    """GS-IR lat-long directions: train.py get_envmap_dirs (centered) or baking.py (corner)."""
    if centered:
        gy = torch.linspace(1.0 / h, 1.0 - 1.0 / h, h, device=device)
        gx = torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=device)
    else:
        gy = torch.linspace(0.0, 1.0 - 1.0 / h, h, device=device)
        gx = torch.linspace(-1.0, 1.0 - 1.0 / w, w, device=device)
    gy, gx = torch.meshgrid(gy, gx, indexing="ij")
    st, ct = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sp, cp = torch.sin(gx * np.pi), torch.cos(gx * np.pi)
    dirs = torch.stack((st * sp, ct, -st * cp), dim=-1)
    if centered:
        return dirs
    d_theta, d_phi = np.pi / h, 2 * np.pi / w
    solid = ((ct - torch.cos(gy * np.pi + d_theta)) * d_phi)[..., None]
    return dirs, solid
