# FSGS — Real-Time Few-shot View Synthesis Using Gaussian Splatting

`fsgs` · Sparse views · Nerfstudio research collection

```bash
pip install -e packages/nerfstudio-methods
ns-train ss-fsgs --data /path/to/scene
```

| Check | Status |
| :--- | :--- |
| Configuration import | passed |
| Short training check | not_run |
| Complete paper reproduction | **Unverified** |
| Historical scene records | 0 (not rerun with this package) |

**Training scope:** Requires monocular depth model weights

**Implementation notes:** Requires monocular depth preprocessing and model weights. The original paper uses a few-view protocol that is not established by RGB smoke tests.

Ten-step checks use a 32×32 analytic sphere fixture, full-resolution training, a fixed seed, and stock Nerfstudio 1.1.5. They exercise model construction, forward/backward, optimizer steps, callbacks, finite parameters, and checkpoint writing. They do not exercise all schedules, prove convergence, or establish the paper's mechanisms. See [the complete validation record](../../results/validation/zoo.json).

[Implementation](../../packages/nerfstudio-methods/splatstudio_zoo/methods/fsgs.py) · [Configurations](../../packages/nerfstudio-methods/splatstudio_zoo/configs.py) · [Source lineage](../../packages/nerfstudio-methods/ORIGIN.json) · [All methods](../methods.md)

## References

- [Reference 1](https://arxiv.org/abs/2312.00451)

References identify the intended technique; they do not certify this port's fidelity. Cite the original authors when using their work.
