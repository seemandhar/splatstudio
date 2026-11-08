"""FSGS density control (C1) and pseudo-camera ownership (C2).

Official `GaussianModel.densify_and_prune` (source/code/scene/gaussian_model.py:497-511), called
from train.py every 100 iterations for 500 < it < 10000 with size_threshold=None:
1. clone: grad >= tau and max scale <= percent_dense * extent;
2. split (post-clone set, clones carry zero gradient): grad >= tau and max scale >
   percent_dense * extent, OR mean squared 3-NN distance > dist_thres * extent and max scale >
   extent (gaussian_model.py:434-439); split parents pruned (it > prune_from_iter);
3. proximity-guided Gaussian unpooling, only while it < 2000 (gaussian_model.py:403-416):
   sources with mean squared 3-NN distance > 5 * extent and max scale > extent grow one new
   Gaussian at the midpoint of each of their 3 edges; new log-scale and opacity logit = the
   destination's, rotation identity, f_dc = f_rest = 0;
4. prune opacity < 0.005 (no screen/world size pruning).
`distCUDA2` (FSGS simple-knn) is an exact 3-NN search excluding the point itself; both rules
need max scale > extent, so the KNN is only evaluated for those Gaussians (torch cdist).
Opacity reset (train.py:166-168): min(o, 0.05) at it > 2000 with (it - 2001) % 3000 == 0.
"""

import numpy as np
import torch
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove, reset_opa, split

from .pseudo import depth_bounds, random_poses_llff

# LLFF poses_bounds.npy near/far of the 3 FSGS training views (min near / max far over the views,
# the only quantities generate_random_poses_llff uses), from ~/datasets/llff/fewview3-r8/<s>/
# prep.json; matched to the scene by the training-camera centres (c2w translation, world units).
SCENE_BOUNDS = {
    "fern": {"near": 17.542520113884287, "far": 78.71925302875819, "train_centers": [
        [-1.658869, -1.609145, -0.242867], [-4.766545, 0.720178, -0.17121],
        [-3.881079, 1.736659, 0.7926]]},
    "flower": {"near": 22.656267550734345, "far": 162.94281555616587, "train_centers": [
        [-4.323173, -2.258927, 1.372147], [-4.035961, 0.720233, -0.023411],
        [-4.668914, 3.036076, -1.819622]]},
    "fortress": {"near": 13.472437825125102, "far": 128.38342286860328, "train_centers": [
        [-2.343452, -2.985312, 1.402754], [-0.596759, 0.169303, -0.406764],
        [4.019429, 2.461339, 0.372861]]},
    "horns": {"near": 12.473701480698228, "far": 168.81094102559487, "train_centers": [
        [-3.77684, -3.147624, -1.43279], [3.351668, 0.198719, 0.574782],
        [3.00166, 2.723918, 0.929056]]},
    "leaves": {"near": 35.262850166998774, "far": 449.9130266665557, "train_centers": [
        [-3.326046, -2.317046, -1.206822], [-1.263424, 0.250338, 0.229431],
        [6.435536, 2.427958, 0.479507]]},
    "orchids": {"near": 15.062411908163723, "far": 96.9429995809968, "train_centers": [
        [-1.975985, -3.74603, -0.495709], [0.042977, 0.292588, -0.050057],
        [0.474495, 3.573621, 0.663743]]},
    "room": {"near": 10.729799505777695, "far": 48.929631822500625, "train_centers": [
        [-3.619974, -3.791045, 0.452161], [0.16049, 2.171683, -0.405748],
        [-0.337692, 1.516422, 0.225718]]},
    "trex": {"near": 26.506352452490084, "far": 112.93942182214286, "train_centers": [
        [-1.90164, -2.832524, 0.605863], [-0.239929, 1.547036, -0.79153],
        [3.713924, 1.138788, -0.674476]]},
}

DEFAULTS = {
    "grad_threshold": 0.0005,
    "min_opacity": 0.005,
    "densification_interval": 100,
    "densify_from_iter": 500,
    "densify_until_iter": 10000,
    "prune_from_iter": 500,
    "percent_dense": 0.01,
    "dist_thres": 10.0,
    "unpool": True,
    "prox_thres": 5.0,
    "prox_k": 3,
    "unpool_until": 2000,
    "opacity_reset_value": 0.05,
    "opacity_reset_interval": 3000,
    "opacity_reset_after": 2000,  # start_sample_pseudo
    "n_pseudo_poses": 10000,
    "pseudo_seed": 0,
    "pseudo_bounds": "official",
    "scene_bounds": SCENE_BOUNDS,
}


def official_bounds(table, viewmats, tol=1e-3):
    """Return (scene, [[near, far]]) whose recorded training-camera centres match these views."""
    centers = np.stack([np.linalg.inv(v)[:3, 3] for v in viewmats])
    for scene, entry in table.items():
        ref = np.asarray(entry["train_centers"], dtype=np.float64)
        if ref.shape == centers.shape and np.abs(ref - centers).max() < tol:
            return scene, np.array([[float(entry["near"]), float(entry["far"])]])
    raise ValueError("pseudo_bounds='official': no scene_bounds entry matches the training "
                     "cameras (add the scene's poses_bounds near/far or set pseudo_bounds='points')")


def knn(query_idx, points, k=3, chunk=1024):
    """Exact k-NN of points[query_idx] among all points, excluding the point itself.

    Returns (mean squared distance [M], neighbour indices [M, k]) like FSGS distCUDA2.
    """
    m = len(query_idx)
    dist = torch.empty(m, device=points.device)
    idx = torch.empty(m, k, dtype=torch.long, device=points.device)
    for s in range(0, m, chunk):
        q = query_idx[s : s + chunk]
        d2 = torch.cdist(points[q], points).square()
        d2[torch.arange(len(q), device=points.device), q] = float("inf")
        vals, nn = d2.topk(k, dim=1, largest=False)
        dist[s : s + chunk] = vals.mean(dim=1)
        idx[s : s + chunk] = nn
    return dist, idx


@torch.no_grad()
def append(params, optimizers, state, new):
    """Append Gaussians with given values; zero Adam moments (official cat_tensors_to_optimizer).

    Parameters are rebuilt from detached tensors with `requires_grad` copied from the old
    Parameter (avoids the frozen-field pitfall in TODO-runtime.md).
    """
    n_new = len(new["means"])

    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p.detach(), new[name].to(p)]),
                                  requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=v.device, dtype=v.dtype)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, torch.zeros((n_new, *v.shape[1:]), device=v.device,
                                                 dtype=v.dtype)))


@torch.no_grad()
def unpool_candidates(params, extent, prox_thres, k=3):
    """Sources (dist > prox_thres * extent, max scale > extent) and their k neighbours."""
    max_scale = params["scales"].exp().max(dim=1).values
    big = torch.where(max_scale > extent)[0]
    if len(big) == 0 or len(params["means"]) <= k:
        empty = torch.zeros(0, dtype=torch.long, device=max_scale.device)
        return empty, torch.zeros(0, k, dtype=torch.long, device=max_scale.device)
    dist, nn = knn(big, params["means"].detach(), k)
    sel = dist > prox_thres * extent
    return big[sel], nn[sel]


@torch.no_grad()
def unpool_values(params, sources, neighbours):
    """New Gaussian values for each (source, destination) edge (gaussian_model.py:408-415)."""
    k = neighbours.shape[1]
    dest = neighbours.reshape(-1)
    src = sources.repeat_interleave(k)
    new = {}
    for name, p in params.items():
        if name == "means":
            new[name] = (p[src] + p[dest]) / 2
        elif name in ("scales", "opacities"):
            new[name] = p[dest].clone()
        elif name == "quats":
            q = torch.zeros_like(p[dest])
            q[:, 0] = 1
            new[name] = q
        else:  # sh0, shN and any extra field: zero
            new[name] = torch.zeros_like(p[dest])
    return new


class FSGSDensity:
    requires = frozenset({"gradient.screen.signed"})

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown FSGS density options: {sorted(unknown)}")
        self.config = {**DEFAULTS, **spec.config}
        if int(self.config["densify_from_iter"]) < int(self.config["prune_from_iter"]):
            raise ValueError("densify_from_iter must be >= prune_from_iter (official keeps split "
                             "parents before prune_from_iter; not implemented)")
        self.state = {"grad2d": None, "count": None}
        self.extent = 1.0
        self.pseudo_info = {}
        self._order = []
        self._rng = np.random.default_rng(int(self.config["pseudo_seed"]))
        self.log = []

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)
        self._init_points = params["means"].detach().cpu().numpy().astype(np.float64)
        self._reset_stats(len(params["means"]), params["means"].device)

    def _reset_stats(self, n, device):
        self.state["grad2d"] = torch.zeros(n, device=device)
        self.state["count"] = torch.zeros(n, device=device)

    # -- pseudo cameras (C2) ------------------------------------------------------------
    def set_training_cameras(self, cameras):
        viewmats = [c["viewmat"].detach().cpu().numpy().astype(np.float64) for c in cameras]
        mode = self.config["pseudo_bounds"]
        if mode == "official":
            scene, bounds = official_bounds(self.config["scene_bounds"], viewmats)
        elif mode == "points":
            Ks = [c["K"].detach().cpu().numpy().astype(np.float64) for c in cameras]
            sizes = [(int(c["width"]), int(c["height"])) for c in cameras]
            scene, bounds = None, depth_bounds(self._init_points, viewmats, Ks, sizes)
        else:
            raise ValueError("pseudo_bounds must be 'official' or 'points'")
        poses, info = random_poses_llff(viewmats, bounds, int(self.config["n_pseudo_poses"]),
                                        self._rng)
        cam0 = cameras[0]
        w, h = int(cam0["width"]), int(cam0["height"])
        # Official PseudoCamera: FoV of the first training view, centred principal point.
        K = cam0["K"].detach().clone()
        K[0, 2], K[1, 2] = w / 2.0, h / 2.0
        self.pseudo_viewmats = torch.tensor(poses, dtype=torch.float32, device=cam0["K"].device)
        self.pseudo_K, self.pseudo_size = K, (w, h)
        self.pseudo_info = {"bounds": bounds.tolist(), "bounds_source": mode, "scene": scene,
                            **info}

    def next_pseudo_camera(self):
        """Official: pop a random element of a copy of the pose list, refill when empty."""
        if not hasattr(self, "pseudo_viewmats"):
            raise RuntimeError("FSGS pseudo views need set_training_cameras (native trainer)")
        if not self._order:
            self._order = list(self._rng.permutation(len(self.pseudo_viewmats)))
        i = int(self._order.pop())
        w, h = self.pseudo_size
        return {"viewmat": self.pseudo_viewmats[i], "K": self.pseudo_K, "width": w, "height": h}

    # -- runtime hooks ----------------------------------------------------------------------
    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()
        outputs["fsgs"] = self

    def accumulate(self, params, info):
        # add_densification_stats: |dL/d(NDC xy)| on visible Gaussians (gsplat pixel scaling).
        means2d = info["means2d"]
        n = len(params["means"])
        if self.state["grad2d"] is None or len(self.state["grad2d"]) != n:
            self._reset_stats(n, params["means"].device)
        if means2d.grad is None:
            return
        grads = means2d.grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        sel = info["radii"] > 0
        if sel.dim() == 3:
            sel = sel.all(dim=-1)
        ids = torch.where(sel)[1]
        self.state["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        self.state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @torch.no_grad()
    def unpool(self, params, optimizers):
        cfg = self.config
        sources, nn = unpool_candidates(params, self.extent, float(cfg["prox_thres"]),
                                        int(cfg["prox_k"]))
        if len(sources) == 0:
            return 0
        new = unpool_values(params, sources, nn)
        append(params, optimizers, self.state, new)
        return len(new["means"])

    @torch.no_grad()
    def densify_and_prune(self, params, optimizers, step):
        cfg, st, extent = self.config, self.state, self.extent
        tau = float(cfg["grad_threshold"])
        pd = float(cfg["percent_dense"])
        grads = st["grad2d"] / st["count"]
        grads[grads.isnan()] = 0.0
        # densify_and_clone
        is_dup = (grads >= tau) & (params["scales"].exp().max(dim=1).values <= pd * extent)
        n_dup = int(is_dup.sum())
        if n_dup:
            duplicate(params, optimizers, st, is_dup)
        # densify_and_split on the post-clone set
        padded = torch.cat([grads, torch.zeros(n_dup, device=grads.device)])
        max_scale = params["scales"].exp().max(dim=1).values
        is_split = (padded >= tau) & (max_scale > pd * extent)
        far = torch.zeros_like(is_split)
        big = torch.where(max_scale > extent)[0]
        if len(big) and len(params["means"]) > 3:
            dist, _ = knn(big, params["means"].detach(), 3)
            far[big[dist > float(cfg["dist_thres"]) * extent]] = True
        is_split |= far
        n_split = int(is_split.sum())
        if n_split:
            split(params, optimizers, st, is_split)
        # proximity-guided unpooling (iter < 2000)
        n_unpool = 0
        if bool(cfg["unpool"]) and step < int(cfg["unpool_until"]):
            n_unpool = self.unpool(params, optimizers)
        is_prune = params["opacities"].sigmoid() < float(cfg["min_opacity"])
        n_prune = int(is_prune.sum()) if step > int(cfg["prune_from_iter"]) else 0
        if n_prune:
            remove(params, optimizers, st, is_prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        row = {"step": step, "cloned": n_dup, "split": n_split, "far_split": int(far.sum()),
               "unpooled": n_unpool, "pruned": n_prune, "gaussians": len(params["means"])}
        self.log.append(row)
        return row

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.config
        if step < int(cfg["densify_until_iter"]):
            self.accumulate(params, outputs["info"])
            if (step > int(cfg["densify_from_iter"])
                    and step % int(cfg["densification_interval"]) == 0):
                self.densify_and_prune(params, optimizers, step)
        after = int(cfg["opacity_reset_after"])
        if step > after and (step - after - 1) % int(cfg["opacity_reset_interval"]) == 0:
            reset_opa(params, optimizers, self.state, value=float(cfg["opacity_reset_value"]))
        for name, p in params.items():
            if not (p.is_leaf and p.requires_grad):
                raise RuntimeError(f"FSGS density left field {name} non-trainable")


def create_density(spec):
    return FSGSDensity(spec)
