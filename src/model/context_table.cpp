#include "model/context_table.h"

#include <stdexcept>

namespace cmix {

namespace {

// 65536 / (n + 1.5): the learning rate for a slot updated n times.
struct RateTable {
  int32_t r[1024];
  RateTable() {
    for (int n = 0; n < 1024; ++n) r[n] = (int32_t)(65536.0 / (n + 1.5));
  }
};
const RateTable kRate;

}  // namespace

ContextTable::ContextTable(int bits) {
  if (bits < 4 || bits > 30) throw std::runtime_error("context table bits must be 4..30");
  slots_.resize(size_t(1) << bits);
  mask_ = (uint64_t(1) << bits) / kBucket - 1;
}

int64_t ContextTable::lookup(uint64_t h) const {
  const size_t b = bucket_of(h);
  const uint16_t c = check_of(h);
  for (int k = 0; k < kBucket; ++k)
    if (slots_[b + k].check == c) return (int64_t)(b + k);
  return -1;
}

uint32_t ContextTable::claim(uint64_t h) {
  const size_t b = bucket_of(h);
  const uint16_t c = check_of(h);
  size_t victim = b;
  for (int k = 0; k < kBucket; ++k) {
    const Slot& s = slots_[b + k];
    if (s.check == c) return (uint32_t)(b + k);
    if (s.check == 0) {
      victim = b + k;
      break;
    }
    if (s.n < slots_[victim].n) victim = b + k;
  }
  slots_[victim] = Slot();
  slots_[victim].check = c;
  return (uint32_t)victim;
}

void ContextTable::update(uint32_t i, int bit, int limit) {
  Slot& s = slots_[i];
  const int target = bit ? 65535 : 0;
  int p = s.p + (int)(((int64_t)(target - s.p) * kRate.r[s.n]) >> 16);
  s.p = (uint16_t)(p < 1 ? 1 : (p > 65535 ? 65535 : p));
  if (s.n < limit) ++s.n;
}

void ContextTable::save(Writer& w) const {
  w.u16_array(reinterpret_cast<const uint16_t*>(slots_.data()), slots_.size() * 4);
}

void ContextTable::load(Reader& r) {
  r.u16_array(reinterpret_cast<uint16_t*>(slots_.data()), slots_.size() * 4);
}

}  // namespace cmix
