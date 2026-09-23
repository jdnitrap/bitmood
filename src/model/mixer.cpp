#include "model/mixer.h"

#include <stdexcept>

#include "core/math.h"

namespace cmix {

int Mixer::mix(const int* p) const {
  int z = 0;
  for (size_t i = 0; i < w_.size(); ++i) z += (w_[i] * stretch(p[i])) >> 7;
  return squash(z);
}

void Mixer::learn(const int* p, int bit, int mixed) {
  int err = (bit ? 4095 : 0) - mixed;
  for (size_t i = 0; i < w_.size(); ++i) {
    int g = (err * stretch(p[i])) >> 16;
    w_[i] = clampi(w_[i] + g, 1, 1024);
  }
}

void Mixer::load(Reader& r) {
  size_t n = w_.size();
  r.vec_i32(w_);
  if (w_.size() != n) throw std::runtime_error("state file: mixer size mismatch");
}

}  // namespace cmix
