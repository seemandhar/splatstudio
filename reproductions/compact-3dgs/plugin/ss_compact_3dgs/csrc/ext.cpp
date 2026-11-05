// Compact-3DGS native port: CUDA hash grid (hashgrid.cu) and a CPU canonical-Huffman decoder.
#include <torch/extension.h>

#include <cstdint>
#include <vector>

torch::Tensor hash_forward(torch::Tensor x, torch::Tensor params, torch::Tensor offsets,
                           torch::Tensor scales, torch::Tensor resolutions);
torch::Tensor hash_backward(torch::Tensor x, torch::Tensor grad_out, int64_t n_params,
                            torch::Tensor offsets, torch::Tensor scales,
                            torch::Tensor resolutions);

// Canonical Huffman decode. bits: uint8 bytes, MSB first. count[l] = number of codes of length l
// (l = 0..maxlen), symbols: symbol ids sorted by (length, symbol) -- the canonical order used by
// codec.canonical_codes. Returns n int64 symbols; raises if the stream is truncated/invalid.
torch::Tensor huffman_decode(torch::Tensor bits, int64_t n, torch::Tensor count,
                             torch::Tensor symbols) {
  TORCH_CHECK(!bits.is_cuda() && bits.scalar_type() == torch::kUInt8 && bits.is_contiguous());
  TORCH_CHECK(!count.is_cuda() && count.scalar_type() == torch::kInt64 && count.is_contiguous());
  TORCH_CHECK(!symbols.is_cuda() && symbols.scalar_type() == torch::kInt64 &&
              symbols.is_contiguous());
  const uint8_t* b = bits.data_ptr<uint8_t>();
  const int64_t nbits = bits.numel() * 8;
  const int64_t* cnt = count.data_ptr<int64_t>();
  const int64_t* sym = symbols.data_ptr<int64_t>();
  const int64_t maxlen = count.numel() - 1;
  auto out = torch::empty({n}, torch::kInt64);
  int64_t* o = out.data_ptr<int64_t>();
  int64_t pos = 0;
  for (int64_t k = 0; k < n; ++k) {
    int64_t code = 0, first = 0, index = 0;
    bool found = false;
    for (int64_t len = 1; len <= maxlen; ++len) {
      TORCH_CHECK(pos < nbits, "huffman stream truncated");
      code |= (b[pos >> 3] >> (7 - (pos & 7))) & 1;
      ++pos;
      const int64_t c = cnt[len];
      if (code - first < c) {
        o[k] = sym[index + code - first];
        found = true;
        break;
      }
      index += c;
      first = (first + c) << 1;
      code <<= 1;
    }
    TORCH_CHECK(found, "invalid huffman code");
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hash_forward", &hash_forward);
  m.def("hash_backward", &hash_backward);
  m.def("huffman_decode", &huffman_decode);
}
