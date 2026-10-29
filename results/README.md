# Evaluation records

| Directory | What the records establish |
| :--- | :--- |
| [native](native) | Fresh short training and held-out rendering runs, complete specifications, source hashes, per-view scores, resources, and traces |
| [validation](validation) | Stock-Nerfstudio imports and ten-step training checks; earlier development findings |
| [historical](historical) | Recovered earlier metric exports with unverified protocols; not rerun results |

Read [the full report](../docs/results.md) for recipes, plots, missing checks, and interpretation. These records do not certify complete paper reproduction. The CSV files are convenience exports; JSON records retain more provenance.

Native checkpoints and datasets are not stored in Git. Published configurations replace private machine paths with dataset descriptors. Rerun the documented commands to regenerate checkpoints and use `ss-eval`. `scripts/publish_results.py` exports measured local runs into this structure without inventing missing metrics.
