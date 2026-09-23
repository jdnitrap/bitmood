#include "model/mixer.h"

#include <stdexcept>

#include "core/math.h"

namespace cmix {

Mixer::Mixer(const std::vector<int>& init, int sets) : n_((int)init.size()), sets_(sets) {
  for (int s = 0; s < sets_; ++s) w_.insert(w_.end(), init.begin(), init.end());
}

int Mixer::mix(const int* p, int set) const {
  const int32_t* w = &w_[(size_t)set * n_];
  int z = 0;
  for (int i = 0; i < n_; ++i) z += (w[i] * stretch(p[i])) >> 7;
  return squash(z);
}

void Mixer::learn(const int* p, int bit, int mixed, int set) {
  int32_t* w = &w_[(size_t)set * n_];
  int err = (bit ? 4095 : 0) - mixed;
  for (int i = 0; i < n_; ++i) {
    int g = (err * stretch(p[i])) >> 16;
    w[i] = clampi(w[i] + g, 1, 1024);
  }
}

void Mixer::load(Reader& r) {
  size_t n = w_.size();
  r.vec_i32(w_);
  if (w_.size() != n) throw std::runtime_error("state file: mixer size mismatch");
}

}  // namespace cmix
