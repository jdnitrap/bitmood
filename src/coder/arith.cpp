#include "coder/arith.h"

#include "core/math.h"

namespace cmix {

namespace {
inline uint32_t split(uint32_t x1, uint32_t x2, int p) {
  return x1 + (uint32_t)(((uint64_t)(x2 - x1) * (uint32_t)clampi(p, kProbMin, kProbMax)) >> 16);
}
}  // namespace

void Encoder::encode(int bit, int p) {
  uint32_t xmid = split(x1_, x2_, p);
  if (bit) x2_ = xmid;
  else x1_ = xmid + 1;
  while (((x1_ ^ x2_) & 0xFF000000u) == 0) {
    out_.push_back((uint8_t)(x2_ >> 24));
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
}

void Encoder::flush() {
  for (int i = 0; i < 4; ++i) {
    out_.push_back((uint8_t)(x2_ >> 24));
    x2_ = (x2_ << 8) + 255;
  }
}

Decoder::Decoder(const uint8_t* p, size_t n) : in_(p), n_(n) {
  for (int k = 0; k < 4; ++k) x_ = (x_ << 8) + (uint32_t)next_byte();
}

int Decoder::decode(int p) {
  uint32_t xmid = split(x1_, x2_, p);
  int bit = x_ <= xmid;
  if (bit) x2_ = xmid;
  else x1_ = xmid + 1;
  while (((x1_ ^ x2_) & 0xFF000000u) == 0) {
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
    x_ = (x_ << 8) + (uint32_t)next_byte();
  }
  return bit;
}

}  // namespace cmix
