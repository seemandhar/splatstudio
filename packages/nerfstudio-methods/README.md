# SplatStudio Nerfstudio methods

33 experimental research configurations imported from SPLATIFY.

```bash
pip install -e packages/nerfstudio-methods
ns-train ss-absgs --data /path/to/scene
```

All 33 configurations import on stock Nerfstudio 1.1.5. Twenty-nine pass a ten-step CUDA training check; four require auxiliary inputs absent from the fixture. These checks do not establish paper reproduction.

See the [method catalog](../../docs/methods.md) for individual commands, paper links, known approximations, and validation status. The `ss-` namespace avoids collisions with an existing SPLATIFY installation. See [ORIGIN.json](ORIGIN.json) for the original source commit and hashes, and [installation](../../docs/installation.md) for the tested environment.
