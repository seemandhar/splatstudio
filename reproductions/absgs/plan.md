# AbsGS plan — method `absgs`, module `ss_absgs`

## Components
| kind | name | factory | requires/provides | config |
|---|---|---|---|---|
| renderer | gsplat (reused built-in) | `splatstudio.backends.gsplat:create_renderer` | provides rgb, depth, gradient.screen.signed, gradient.screen.absolute | `absgrad=True` |
| density | absgs-density | `ss_absgs.density:create_density` | requires gradient.screen.signed + gradient.screen.absolute | clone_grad2d 0.0002, split_grad2d 0.0004, grow_scale3d 0.001, refine 500/15000/100, reset 3000, prune_opa 0.005, prune_scale3d 0.1, reduce_every 500 |
| loss | photometric (reused) | `splatstudio.losses:create_photometric` | render.rgb | L1 + 0.2 D-SSIM |

`AbsGSStrategy(DefaultStrategy)` keeps two accumulators (`grad2d` signed norm, `grad2d_abs`
norm of `means2d.absgrad`), same NDC scaling (W/2, H/2) and visibility mask (`radii>0`) as gsplat,
and replaces `_grow_gs`: clone = signed ≥ τ_clone & small; split = abs ≥ τ_split & large.
Both accumulators are N-sized state tensors, so gsplat ops keep them aligned. Pruning and reset
are inherited unchanged. `reduce_every=500` (official `use_reduce` default, on) clamps opacity to ≤0.8 and zeroes Adam moments every 500 steps before 15k; `absgs-noreduce` disables it.

No new fields. No custom CUDA (gsplat `absgrad` is the identical per-pixel |·| accumulation as
the official `diff-gaussian-rasterization-abs` backward.cu:549-550).

## Probes
| probe | contribution | fails if |
|---|---|---|
| homodirectional-gradient (CUDA) | C1 | gsplat absgrad ≠ independent Σ_pixels |∂L_j/∂μ| computed by per-pixel autograd; or no collision (|Σ| ≈ Σ|·|) on the fixture |
| split-clone-criteria | C2 | clone uses abs grad, split uses signed grad, wrong τ_clone/τ_split/τ_S, accumulation normalization wrong (fixture vs. hand-computed decisions; also asserts gsplat-absgrad's decisions differ) |
| state-transitions, photometric-gradient, renderer-gradient | infra | built-in |

## Fidelity gaps
1. Runtime protocol differences shared by all methods (white bg, random-with-replacement
   sampling, SSIM valid conv, LPIPS package) — compare improvement over our gsplat baseline.
2. Our gsplat baseline uses τ_S=0.01; paper's improvement over "3D-GS" (paper numbers, τ_S=0.01)
   is the right comparison; part of AbsGS's indoor gain comes from τ_S (3D-GS* row).
3. gsplat 3DGS screen-size pruning (max_radii2D>20) absent (refine_scale2d_stop_iter=0); the
   official code's radii prune is effectively inactive too (README notes the bug). Small.
4. The official README's "not used in our paper" extras (initial SfM filtering + init_prune, weight pruning) are off. Periodic opacity reduction is NOT in that list and is on by default in the release (arguments:92-93, train.py:132, full_eval.py), so `absgs` enables it (review 1); the paper never mentions it, so whether the table used it is unknown (`absgs-noreduce` keeps the alternative).
5. Review-1 minors: released loader filters SfM points when points3D.ply is absent (we keep all, per README); clone-not-split padding unprobed (sets disjoint by scale); runtime densifies after optimizer.step (shared with baseline); white-bg alpha term enters both gradient accumulators (shared protocol gap).
