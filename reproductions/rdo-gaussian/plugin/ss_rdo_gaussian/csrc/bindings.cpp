/* Python bindings for the vendored rANS coder (one torch cpp_extension module).
 *
 * rans_interface.{hpp,cpp} and ops.cpp are copied unchanged (apart from removing their own
 * PYBIND11_MODULE blocks) from the RDO-Gaussian official repository
 * (submodules/imcl-compression/data_compression/cpp_exts, commit 710ebab8), which in turn are
 * CompressAI's rANS interface (Copyright 2020 InterDigital Communications, Inc., Apache-2.0).
 * rans64.h is Fabian Giesen's ryg_rans (CC0, see LICENSE-ryg_rans).
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

#include "rans_interface.hpp"

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float> &pmf, int precision);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "RDO-Gaussian rANS entropy coder (CompressAI / ryg_rans)";
  m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf, "Return quantized CDF for a given PMF");
  py::class_<RansEncoder>(m, "RansEncoder")
      .def(py::init<>())
      .def("encode_with_indexes", &RansEncoder::encode_with_indexes);
  py::class_<RansDecoder>(m, "RansDecoder")
      .def(py::init<>())
      .def("decode_with_indexes", &RansDecoder::decode_with_indexes);
}
