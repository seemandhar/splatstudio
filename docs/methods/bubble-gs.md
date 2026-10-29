# BubbleGS: Physics-Informed Gaussian Splatting for Iridescent Thin Films

`bubble-gs` · Exploratory · Nerfstudio research collection

```bash
pip install -e packages/nerfstudio-methods
ns-train ss-bubble-gs --data /path/to/scene
```

| Check | Status |
| :--- | :--- |
| Configuration import | passed |
| Short training check | passed |
| Complete paper reproduction | **Unverified** |
| Historical scene records | 0 (not rerun with this package) |

**Training scope:** Short training with callbacks and finite-state checks; late schedules untested

**Implementation notes:** Experimental scene-specific extension. Physical accuracy and scientific novelty are not established.

Ten-step checks use a 32×32 analytic sphere fixture, full-resolution training, a fixed seed, and stock Nerfstudio 1.1.5. They exercise model construction, forward/backward, optimizer steps, callbacks, finite parameters, and checkpoint writing. They do not exercise all schedules, prove convergence, or establish the paper's mechanisms. See [the complete validation record](../../results/validation/zoo.json).

[Implementation](../../packages/nerfstudio-methods/splatstudio_zoo/methods/bubble_gs.py) · [Configurations](../../packages/nerfstudio-methods/splatstudio_zoo/configs.py) · [Source lineage](../../packages/nerfstudio-methods/ORIGIN.json) · [All methods](../methods.md)

## References

No standalone paper reference is asserted for this configuration. Consult the source lineage.

References identify the intended technique; they do not certify this port's fidelity. Cite the original authors when using their work.
