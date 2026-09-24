// Logistic mixer: p = squash(sum of w_i * x_i), where x_i are the
// specialists' votes in stretch space. Weights are floats and may go
// negative (a specialist that is reliably wrong becomes useful inverted).
//
// A mixer holds several weight sets; the caller picks one per bit with a
// selector (for example match state, or which grid view is winning), so
// each situation learns its own trust in each specialist.
#pragma once

#include <vector>

#include "core/serial.h"

namespace cmix {

class Mixer {
 public:
  Mixer(int inputs, int sets, float init_weight);
  int inputs() const { return n_; }
  int sets() const { return sets_; }
  // Dot product in stretch space (before squash).
  float dot(const float* x, int set) const;
  // err = bit - p (p in 0..1). lr is the learning rate.
  void learn(const float* x, int set, float err, float lr);
  void set(int input, int set, float weight);
  // Copy one weight set onto another (a new region starts from the current trust).
  void copy_set(int from, int to);
  float weight(int i, int set) const { return w_[(size_t)set * n_ + i]; }
  void save(Writer& w) const { w.vec_f32(w_); }
  void load(Reader& r);

 private:
  int n_, sets_;
  std::vector<float> w_;
};

}  // namespace cmix
