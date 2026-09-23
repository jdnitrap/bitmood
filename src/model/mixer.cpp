#include "model/mixer.h"

#include <stdexcept>

#include "core/math.h"

namespace cmix {

Mixer::Mixer(int inputs, int sets, float init_weight)
    : n_(inputs), sets_(sets), w_((size_t)inputs * sets, init_weight) {}

float Mixer::dot(const float* x, int set) const {
  const float* w = &w_[(size_t)set * n_];
  float z = 0;
  for (int i = 0; i < n_; ++i) z += w[i] * x[i];
  return z;
}

void Mixer::learn(const float* x, int set, float err, float lr) {
  float* w = &w_[(size_t)set * n_];
  const float g = err * lr;
  for (int i = 0; i < n_; ++i) w[i] = clampf(w[i] + g * x[i], -16.0f, 16.0f);
}

void Mixer::load(Reader& r) {
  size_t n = w_.size();
  r.vec_f32(w_);
  if (w_.size() != n) throw std::runtime_error("state file: mixer size mismatch");
}

}  // namespace cmix
