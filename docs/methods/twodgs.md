# 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

`twodgs` · Geometry · Nerfstudio research collection

```bash
pip install -e packages/nerfstudio-methods
ns-train ss-twodgs --data /path/to/scene
```

| Check | Status |
| :--- | :--- |
| Configuration import | passed |
| Short training check | passed |
| Complete paper reproduction | **Unverified** |
| Historical scene records | 0 (not rerun with this package) |

**Training scope:** Short training with callbacks and finite-state checks; late schedules untested

**Implementation notes:** Legacy approximation: this port uses the 3D rasterizer and geometry regularizers, not perspective-correct 2D ray-splat intersections. Use gsplat-surfel to exercise the actual 2D rasterizer.

Ten-step checks use a 32×32 analytic sphere fixture, full-resolution training, a fixed seed, and stock Nerfstudio 1.1.5. They exercise model construction, forward/backward, optimizer steps, callbacks, finite parameters, and checkpoint writing. They do not exercise all schedules, prove convergence, or establish the paper's mechanisms. See [the complete validation record](../../results/validation/zoo.json).

[Implementation](../../packages/nerfstudio-methods/splatstudio_zoo/methods/twodgs.py) · [Configurations](../../packages/nerfstudio-methods/splatstudio_zoo/configs.py) · [Source lineage](../../packages/nerfstudio-methods/ORIGIN.json) · [All methods](../methods.md)

## References

- [Reference 1](https://surfsplatting.github.io)

References identify the intended technique; they do not certify this port's fidelity. Cite the original authors when using their work.
