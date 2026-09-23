#include "model/match_model.h"

#include <algorithm>

#include "core/hash.h"
#include "core/math.h"

namespace cmix {

namespace {
constexpr uint32_t kMaxLen = 65535;
constexpr int kBuckets = 32;
}  // namespace

MatchModel::MatchModel(int index_bits)
    : bits_(index_bits), index_(size_t(1) << index_bits, 0), map_(kBuckets * 2, 1023) {}

uint32_t MatchModel::key(const History& h) const {
  uint64_t x = 0x5EED;
  for (int i = 0; i < kMinLen; ++i) x = hash_add(x, h.back((size_t)i));
  return (uint32_t)(x >> (64 - bits_));
}

void MatchModel::advance(const History& h, Stream& s) {
  const size_t n = h.size();
  const uint8_t b = h.back(0);
  // Follow the current match, or drop it.
  if (s.match_len > 0) {
    if (h.has(s.match_ptr) && s.match_ptr + 1 < h.end() && h.at(s.match_ptr) == b) {
      ++s.match_ptr;
      s.match_len = std::min(s.match_len + 1, kMaxLen);
    } else {
      s.match_len = 0;
    }
  }
  if (n < (size_t)kMinLen) return;
  const uint32_t k = key(h);
  if (s.match_len == 0 && index_[k] != 0) {
    // Candidate: index of the byte that followed the same kMinLen bytes before.
    const size_t cand = index_[k] - 1;
    if (cand < n) {
      uint32_t len = 0;
      while (len < kMaxLen && len < cand && h[cand - 1 - len] == h[n - 1 - len]) ++len;
      if (len >= (uint32_t)kMinLen) {
        s.match_len = len;
        s.match_ptr = h.base() + cand;
      }
    }
  }
  index_[k] = (uint32_t)n + 1;  // this position follows these kMinLen bytes
}

void MatchModel::rebuild(const History& h) {
  std::fill(index_.begin(), index_.end(), 0);
  const size_t n = h.size();
  for (size_t end = (size_t)kMinLen; end <= n; ++end) {
    uint64_t x = 0x5EED;
    for (int i = 0; i < kMinLen; ++i) x = hash_add(x, h[end - 1 - (size_t)i]);
    index_[(uint32_t)(x >> (64 - bits_))] = (uint32_t)end + 1;
  }
}

int MatchModel::expected(const History& h, const Stream& s, BitPos bp) {
  if (s.match_len == 0 || !h.has(s.match_ptr)) return -1;
  const int e = h.at(s.match_ptr);
  if (((e | 0x100) >> (8 - bp.index)) != (int)bp.key()) return -1;
  return e;
}

int MatchModel::length_bucket(uint32_t len) {
  if (len < 16) return (int)len;
  return std::min(kBuckets - 1, 16 + (int)((len - 16) / 8));
}

int MatchModel::state(const History& h, const Stream& s, BitPos bp) {
  if (expected(h, s, bp) < 0) return 0;
  return s.match_len < 16 ? 1 : (s.match_len < 32 ? 2 : 3);
}

void MatchModel::predict(const History& h, const Stream& s, BitPos bp, float* x, int* p, int& slot) const {
  const int e = expected(h, s, bp);
  if (e < 0) {
    x[0] = x[1] = 0.0f;
    p[0] = p[1] = 32768;
    slot = -1;
    return;
  }
  const int bit = (e >> (7 - bp.index)) & 1;
  slot = length_bucket(s.match_len) * 2 + bit;
  p[0] = map_.p((size_t)slot);
  x[0] = stretch(p[0]);
  // Fixed-strength vote: grows with match length.
  const float strength = std::min(32u, s.match_len) / 4.0f;
  x[1] = bit ? strength : -strength;
  p[1] = to_p16(squash(x[1]));
}

void MatchModel::learn(int slot, int bit) {
  if (slot >= 0) map_.update((size_t)slot, bit);
}

}  // namespace cmix
