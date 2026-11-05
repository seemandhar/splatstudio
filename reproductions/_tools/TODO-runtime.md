- [ ] splatstudio.runtime.knn_log_scales: chunk size must adapt to N (4096×4M cdist = 61 GiB).
      Fix only when no runs are in flight (runs hash runtime source). EDGS used --recipe splatstudio as a workaround.
- [ ] Component state persistence: components get state_dict()/load_state_dict(); runtime.train saves them in the
      checkpoint and runtime.evaluate restores them before rendering. Needed by MLP/anchor methods (HAC #41 packs MLP
      weights into a per-anchor field `hac_blob` as a workaround; ContextGS #44, Scaffold-GS family). Design note:
      reproductions/hac/plan.md. Apply only when no runs are in flight.
- [ ] View sampling: offer the official 3DGS shuffled-stack sampler (each training view once per epoch) as a trainer option.
      With few-view (3 views) sampling-with-replacement changes per-view visit counts ±15% per 100-step window (AD-GS #34 analysis).

## gsplat backward drops gradients for saturated alpha (found via DropGaussian, 2026-11-04)
gsplat `RasterizeToPixels3DGSBwd.cu:222` only back-propagates opacity/conic/means2d gradients
where `opac * G <= 0.999`; the Inria rasterizer clamps alpha (0.99) in the forward pass but always
back-propagates. Methods whose effective opacity can exceed ~1 (DropGaussian compensation, any
opacity scaling) silently lose gradients. Plugin fix: straight-through cap at 0.999 (see
reproductions/dropgaussian/plugin/ss_dropgaussian/renderer.py). Consider a runtime-wide option.

## Inria screen-size prune is a no-op in official 3DGS-derived code (found via SparseGS, 2026-11-04)
Official `densification_postfix` (called by clone and split) zeroes `max_radii2D` before
`densify_and_prune` checks `max_radii2D > max_screen_size`, so the screen-size prune never fires;
only the world-size prune (scale > 0.1·extent) acts. Ports that implement a real screen-size prune
over-prune (SparseGS 12-view: test PSNR 18.3 -> 12.2 after step 3000). Check other ports that copied
the Inria density (`max_screen_size` / `size_threshold = 20`).
