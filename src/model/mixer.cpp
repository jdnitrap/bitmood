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

void Mixer::set(int input, int set, float weight) { w_[(size_t)set * n_ + input] = weight; }

void Mixer::copy_set(int from, int to) {
  if (from < 0 || to < 0 || from >= sets_ || to >= sets_)
    throw std::runtime_error("mixer weight set out of range");
  const float* src = &w_[(size_t)from * n_];
  float* dst = &w_[(size_t)to * n_];
  for (int i = 0; i < n_; ++i) dst[i] = src[i];
}

void Mixer::load(Reader& r) {
  size_t n = w_.size();
  r.vec_f32(w_);
  if (w_.size() != n) throw std::runtime_error("state file: mixer size mismatch");
}

}  // namespace cmix
