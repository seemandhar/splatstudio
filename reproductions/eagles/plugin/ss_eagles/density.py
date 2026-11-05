"""EAGLES density component: decoder bank + 3DGS densification + influence pruning + codec.

Wraps gsplat's DefaultStrategy with the official `configs/efficient-3dgs.yaml` schedule
(densify every 125 steps from 500 to 18000, opacity reset every 2500 below 18000, influence
pruning of the 15% least influential Gaussians every 5100 steps below 27000) and owns the
EAGLES decoders (global learned state the runtime cannot hold). Per step, after the runtime's
per-Gaussian optimizer step (official order inside an iteration: influence stats -> densify ->
opacity reset -> influence prune -> optimizer step):

1. decoder Adam step (gradients from the same backward as the latents);
2. mirror D(STE(q)) into the base fields quats / opacities / shN, so gsplat's split rotation,
   opacity pruning and all checks see the decoded attributes;
3. accumulate the influence sum_p alpha*T of this render (CUDA, step < 27000);
4. DefaultStrategy refine/prune; Gaussians created by clone/split start with influence 0
   (official densification_postfix);
5. opacity reset on the opacity latent: q = D^{-1}(logit(min(sigmoid(D(q)), 0.01))), Adam
   moments zeroed (official reset_opacity / replace_tensor_to_optimizer);
6. influence pruning (W <= quantile(W, 0.15)), influence reset;
7. latent LRs from the decoder norms (official update_learning_rate), next progressive scale;
8. at total_steps: round -> entropy-encode -> decode the bitstream -> decoders -> bake into the
   base fields (bank.baked), so validation, the checkpoint and `ss-eval` measure the decoded model.
"""

import json
import math
import time

import torch
from gsplat import DefaultStrategy
from gsplat.strategy import ops

from . import codec, decoders, kernels

INFLUENCE = "eagles_influence"

DEFAULTS = {
    "total_steps": 30000,
    "base": {"absgrad": False, "grow_grad2d": 0.0002, "refine_start_iter": 500,
             "refine_stop_iter": 18000, "refine_every": 125, "reset_every": 2500},
    "opacity_reset_every": 2500,
    "opacity_reset_until": 18000,
    "opacity_reset_value": 0.01,
    "influence_prune": True,
    "infl_prune_every": 5100,
    "infl_prune_until": 27000,
    "infl_quantile": 0.15,
    "progressive": True,
    "resize_scale": 0.3,
    "resize_period": 0.7,
    "attributes": decoders.ATTRIBUTES,
    "decoder_seed": 0,
    "encode": True,
}


def resize_scale(step, cfg):
    """Official DecayScheduler('cosine', total_steps=int(period*(iterations+1)), 0.3 -> 1.0)."""
    if not cfg["progressive"]:
        return 1.0
    total = int(cfg["resize_period"] * (cfg["total_steps"] + 1))
    start, end = float(cfg["resize_scale"]), 1.0
    if step > total:
        return end
    return end + 0.5 * (start - end) * (1 + math.cos(step / total * math.pi))


def schedule_events(step, cfg):
    """EAGLES events after optimizer step `step` (1-based), excluding gsplat refine."""
    events = []
    if step < cfg["opacity_reset_until"] and step % cfg["opacity_reset_every"] == 0:
        events.append("opacity_reset")
    if (cfg["influence_prune"] and step % cfg["infl_prune_every"] == 0
            and step < cfg["infl_prune_until"]):
        events.append("influence_prune")
    if cfg["encode"] and step == cfg["total_steps"]:
        events.append("encode")
    return events


class EaglesStrategy(DefaultStrategy):
    """DefaultStrategy whose clones/split children start with zero accumulated influence."""

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        n_dupli, n_split = super()._grow_gs(params, optimizers, state, step)
        # duplicate appends n_dupli copies; split removes the n_split parents and appends
        # 2 n_split children, so the newest n_dupli + 2 n_split rows are the new Gaussians.
        n_new = n_dupli + 2 * n_split
        if n_new and isinstance(state.get(INFLUENCE), torch.Tensor):
            state[INFLUENCE][-n_new:] = 0.0
        return n_dupli, n_split


class EaglesDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown eagles-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.strategy = EaglesStrategy(**self.cfg["base"])
        self.requires = {"gradient.screen.absolute" if self.strategy.absgrad
                         else "gradient.screen.signed"}
        self.bank = None
        self.strategy_state = None
        self.history = []
        self.encoded = None
        self.stream = None

    @property
    def state(self):
        return {"strategy": self.strategy_state,
                "eagles": {"events": self.history, "encoded": self.encoded,
                           "decoders": self.bank.state_tensors() if self.bank else None,
                           "bitstream": self.stream}}

    def initialize(self, params, optimizers, scene_scale=1.0):
        missing = [f for f in ("q_shN", "q_rot", "q_opa") if f not in params]
        if missing:
            raise ValueError(f"eagles-density needs the latent fields {missing}")
        self.strategy.check_sanity(params, optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)
        device = params["means"].device
        self.bank = decoders.DecoderBank(self.cfg["attributes"], self.cfg["decoder_seed"])
        self.bank.to(device)
        self.bank.build_optimizer()
        n = len(params["means"])
        init = self.bank.initial_latents(n, params["quats"].detach(), params["opacities"].detach())
        with torch.no_grad():
            for name, value in init.items():
                if params[name].shape != value.shape:
                    raise ValueError(f"{name}: latent dimension differs from the decoder")
                params[name].copy_(value)
        self.strategy_state[INFLUENCE] = torch.zeros(n, device=device)
        self._set_latent_lrs(optimizers)
        # No mirroring here: the base fields may already sit in an autograd graph (a render
        # before initialize); the renderer decodes the latents, mirrors follow each step.
        self.bank.scale = resize_scale(1, self.cfg)
        decoders.ACTIVE["bank"] = self.bank

    def before_backward(self, params, optimizers, step, outputs):
        self.strategy.step_pre_backward(params, optimizers, self.strategy_state, step,
                                        outputs["info"])

    def after_backward(self, params, optimizers, step, outputs):
        bank = self.bank
        bank.step()
        self._mirror(params)
        if self.cfg["influence_prune"] and step < self.cfg["infl_prune_until"]:
            w = kernels.influence(outputs["info"])
            self.strategy_state[INFLUENCE] += w
        self.strategy.step_post_backward(params, optimizers, self.strategy_state, step,
                                         outputs["info"], packed=False)
        for event in schedule_events(step, self.cfg):
            start = time.perf_counter()
            record = {"eagles": event, "step": step, "before": len(params["means"])}
            if event == "opacity_reset":
                self._reset_opacity(params, optimizers)
            elif event == "influence_prune":
                record.update(self._influence_prune(params, optimizers))
            elif event == "encode":
                record.update(self._encode(params))
            record.update(after=len(params["means"]),
                          seconds=round(time.perf_counter() - start, 2))
            if event != "opacity_reset":
                self.history.append(record)
                print(json.dumps(record), flush=True)
        if not bank.baked:
            self._mirror(params)
            self._set_latent_lrs(optimizers)
            bank.scale = resize_scale(step + 1, self.cfg)

    @torch.no_grad()
    def _mirror(self, params):
        for field, value in self.bank.decode_all(params).items():
            params[field].copy_(value.reshape(params[field].shape))

    def _set_latent_lrs(self, optimizers):
        for name in self.bank.attributes:
            field = decoders.TARGETS[name][1]
            optimizers[field].param_groups[0]["lr"] = self.bank.latent_lr(name)

    @torch.no_grad()
    def _reset_opacity(self, params, optimizers):
        """Official reset_opacity: inverse_sigmoid(min(sigmoid(D(q)), 0.01)) -> D^{-1}."""
        value = float(self.cfg["opacity_reset_value"])
        op = torch.sigmoid(self.bank.decode(params, "opa"))
        x = torch.min(op, torch.ones_like(op) * value)
        logits = torch.log(x / (1 - x))  # official inverse_sigmoid
        latent = self.bank.decoders["opa"].invert(logits.reshape(-1, 1))

        def param_fn(name, p):
            return torch.nn.Parameter(latent.reshape(p.shape).to(p.dtype),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros_like(v)

        ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers,
                                         names=["q_opa"])

    @torch.no_grad()
    def _influence_prune(self, params, optimizers):
        w = self.strategy_state[INFLUENCE]
        threshold = torch.quantile(w, float(self.cfg["infl_quantile"]))
        mask = w <= threshold
        n_pruned = int(mask.sum())
        if n_pruned:
            ops.remove(params, optimizers, self.strategy_state, mask)
        self.strategy_state[INFLUENCE].zero_()
        return {"pruned": n_pruned, "threshold": float(threshold)}

    @torch.no_grad()
    def _encode(self, params):
        bank = self.bank
        n = len(params["means"])
        latents = {f: torch.round(params[f].detach()).reshape(n, -1) for f in bank.fields}
        t0 = time.perf_counter()
        size = codec.paper_size_bits(latents, bank, n)
        stream = codec.encode(latents)
        t1 = time.perf_counter()
        decoded_int = codec.decode(stream)
        t2 = time.perf_counter()
        for f, q in latents.items():
            if not torch.equal(decoded_int[f].to(q.device), q):
                raise RuntimeError(f"{f}: entropy decode does not reproduce the latents")
        for f, q in decoded_int.items():
            params[f].copy_(q.to(params[f].device).reshape(params[f].shape))
        self._mirror(params)  # decoders on the decoded integers: the decoded model
        bank.baked = True
        actual = codec.actual_bytes(stream, bank, n)
        self.stream = stream
        self.encoded = {
            "gaussians": n,
            "paper_size_MB": size["total_bits"] / 8 / 1e6,
            "paper_size_bits": size,
            "actual_MB": actual["total_bytes"] / 1e6,
            "actual_bytes": actual,
            "latent_bits_per_gaussian": {f: b / n for f, b in size["latent_bits"].items()},
            "uncompressed_3dgs_ply_MiB": n * 62 * 4 / 2**20,
            "encode_seconds": round(t1 - t0, 2),
            "decode_seconds": round(t2 - t1, 2),
        }
        return {"paper_size_MB": round(self.encoded["paper_size_MB"], 3),
                "actual_MB": round(self.encoded["actual_MB"], 3)}


def create_density(spec):
    return EaglesDensity(spec)
