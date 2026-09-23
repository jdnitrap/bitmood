// Logistic mixer: weighted sum of stretched specialist votes, squashed back
// to a probability. Weights are learned online, one small step per bit.
#pragma once

#include <vector>

#include "core/serial.h"

namespace cmix {

class Mixer {
 public:
  explicit Mixer(int n) : w_(n, 128) {}
  int size() const { return (int)w_.size(); }
  int mix(const int* p) const;
  void learn(const int* p, int bit, int mixed);
  int weight(int i) const { return w_[i]; }
  void save(Writer& w) const { w.vec_i32(w_); }
  void load(Reader& r);

 private:
  std::vector<int32_t> w_;  // 128 = weight 1.0, kept in 1..1024
};

}  // namespace cmix
