#include "model/specialists.h"

#include <algorithm>

#include "core/hash.h"
#include "core/math.h"

namespace cmix {

CountTable::CountTable(int bits, int limit)
    : mask_((1u << bits) - 1), limit_(limit), n0_(size_t(1) << bits, 1), n1_(size_t(1) << bits, 1) {}

int CountTable::predict(uint32_t ctx) const {
  uint32_t i = ctx & mask_;
  int a = n0_[i], b = n1_[i];
  return clampi((b * 4095) / (a + b), 1, 4094);
}

void CountTable::learn(uint32_t ctx, int bit) {
  uint32_t i = ctx & mask_;
  if (bit) n1_[i] = (uint16_t)std::min(n1_[i] + 1, 65535);
  else n0_[i] = (uint16_t)std::min(n0_[i] + 1, 65535);
  if (n0_[i] + n1_[i] > limit_) {
    n0_[i] = (uint16_t)((n0_[i] + 1) >> 1);
    n1_[i] = (uint16_t)((n1_[i] + 1) >> 1);
  }
}

void CountTable::save(Writer& w) const {
  w.vec_u16(n0_);
  w.vec_u16(n1_);
}

void CountTable::load(Reader& r) {
  size_t n = n0_.size();
  r.vec_u16_exact(n0_, n);
  r.vec_u16_exact(n1_, n);
}

uint32_t RecentPattern::ctx(const History& h, const Stream& s, BitPos bp) {
  uint32_t c = hash_mix(0xA5A5A5A5u, (uint32_t)s.last_byte);
  c = hash_mix(c, (uint32_t)bp.index);
  c = hash_mix(c, (uint32_t)bp.partial);
  if (h.size() >= 1) c = hash_mix(c, h.back(0));
  if (h.size() >= 2) c = hash_mix(c, h.back(1));
  return c;
}

int LongMatch::predict(const History& h, BitPos bp) const {
  const size_t pos = h.size();
  if (pos < 8) return 2048;
  const int have = 8 - bp.index;
  const int mask = (0xFF << have) & 0xFF;
  const int want = (bp.partial << have) & mask;
  int best = 0;
  int pred = 0;
  for (size_t off = 3; off < pos && off < 4096; ++off) {
    size_t match = 0;
    while (match < 64 && match < pos - off && h[pos - off - 1 - match] == h[pos - 1 - match]) ++match;
    if ((int)match <= best) continue;
    // The candidate must agree with the bits already decided in this byte.
    uint8_t nxt = h[pos - off];
    if ((nxt & mask) != want) continue;
    best = (int)match;
    pred = (nxt >> (7 - bp.index)) & 1;
  }
  if (best < 3) return 2048;
  int conf = std::min(1800, best * 80);
  return pred ? 2048 + conf : 2048 - conf;
}

int ByteShape::predict(int last_byte, BitPos bp) const {
  int p = 2048;
  if (last_byte < 0) return p;
  const int k = bp.index;
  if (is_alpha(last_byte)) {
    if (k == 0) p -= 400;
    if (k == 1) p += 350;
    if (k == 2 && (bp.partial & 2)) p += 80;
  }
  if (is_digit(last_byte)) {
    if (k == 0) p -= 500;
    if (k == 1) p += 200;
    if (k == 2) p += 400;
    if (k == 3) p += 400;
  }
  if (is_space(last_byte) && k == 0) p -= 300;
  if (is_punct(last_byte) && k == 0) p -= 250;
  if (utf8_lead(last_byte) && k == 0) p += 200;
  return clampi(p, 1, 4094);
}

}  // namespace cmix
