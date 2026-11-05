Vendored entropy coder of the official RDO-Gaussian code
(https://github.com/USTC-IMCL/RDO-Gaussian, `submodules/imcl-compression`, commit 710ebab8):

* `rans_interface.hpp`, `rans_interface.cpp`, `ops.cpp`: CompressAI rANS interface and
  `pmf_to_quantized_cdf`, Copyright 2020 InterDigital Communications, Inc., Apache License 2.0.
  Copied verbatim except that their `PYBIND11_MODULE` blocks were removed; `bindings.cpp`
  exposes them as one torch cpp_extension module.
* `rans64.h`: ryg_rans by Fabian Giesen, CC0 (`LICENSE-ryg_rans`).

Built on first use (CPU only) by `ss_rdo_gaussian/rans.py` via `torch.utils.cpp_extension.load`.
