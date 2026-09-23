#include "model/adaptive.h"

#include <cmath>

#include "core/math.h"

namespace cmix {

AdaptiveMap::AdaptiveMap(size_t n, int limit) : limit_(limit), p_(n, 32768), n_(n, 0) {}

void AdaptiveMap::update(size_t i, int bit) {
  const int target = bit ? 65535 : 0;
  const double rate = 1.0 / (n_[i] + 1.5);
  int p = p_[i] + (int)std::lround((target - p_[i]) * rate);
  p_[i] = (uint16_t)clampi(p, 1, 65535);
  if (n_[i] < limit_) ++n_[i];
}

void AdaptiveMap::save(Writer& w) const {
  w.u16_array(p_.data(), p_.size());
  w.u16_array(n_.data(), n_.size());
}

void AdaptiveMap::load(Reader& r) {
  r.u16_array(p_.data(), p_.size());
  r.u16_array(n_.data(), n_.size());
}

// Points j = 0..32 sit at stretch = (j - 16) / 2, covering -8..8.
Apm::Apm(size_t contexts, int rate_bits) : contexts_(contexts), rate_(rate_bits), t_(contexts * 33) {
  for (size_t c = 0; c < contexts; ++c)
    for (int j = 0; j < 33; ++j) t_[c * 33 + j] = (uint16_t)to_p16(squash((j - 16) / 2.0f));
}

int Apm::refine(int p, size_t ctx, uint32_t& slot) const {
  const float s = clampf((stretch(p) + 8.0f) * 2.0f, 0.0f, 31.999f);
  const int j = (int)s;
  const float w = s - (float)j;
  const size_t base = ctx * 33 + (size_t)j;
  slot = (uint32_t)(w < 0.5f ? base : base + 1);  // update the nearer point
  return clampi((int)std::lround(t_[base] * (1.0f - w) + t_[base + 1] * w), 1, 65535);
}

void Apm::update(uint32_t slot, int bit) {
  const int target = bit ? 65535 : 0;
  int v = t_[slot] + ((target - t_[slot]) >> rate_);
  t_[slot] = (uint16_t)clampi(v, 1, 65535);
}

}  // namespace cmix
