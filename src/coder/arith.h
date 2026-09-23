// Binary arithmetic coder (FPAQ0 style, carryless, 32-bit range).
// p is P(bit = 1) as a 16-bit value in 1..65535; bit 1 takes the lower
// piece of the range, of size p / 65536.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cmix {

class Encoder {
 public:
  explicit Encoder(std::vector<uint8_t>& out) : out_(out) {}
  void encode(int bit, int p);
  void flush();

 private:
  std::vector<uint8_t>& out_;
  uint32_t x1_ = 0, x2_ = 0xFFFFFFFFu;
};

class Decoder {
 public:
  // Reads from [p, p + n). Past the end it reads zeros.
  Decoder(const uint8_t* p, size_t n);
  int decode(int p);

 private:
  int next_byte() { return pos_ < n_ ? in_[pos_++] : 0; }
  const uint8_t* in_;
  size_t n_, pos_ = 0;
  uint32_t x1_ = 0, x2_ = 0xFFFFFFFFu, x_ = 0;
};

}  // namespace cmix
