// Logistic mixer: weighted sum of stretched specialist votes, squashed back
// to a probability. Weights are learned online, one small step per bit.
//
// A mixer can hold several weight sets; the caller picks one per bit
// (for example "which grid view is winning right now"), so different
// regions of a file learn different trust in each specialist.
#pragma once

#include <vector>

#include "core/serial.h"

namespace cmix {

class Mixer {
 public:
  // init[i] is the starting weight of input i in every set (128 = 1.0).
  Mixer(const std::vector<int>& init, int sets = 1);
  int size() const { return n_; }
  int sets() const { return sets_; }
  int mix(const int* p, int set) const;
  void learn(const int* p, int bit, int mixed, int set);
  int weight(int i, int set = 0) const { return w_[(size_t)set * n_ + i]; }
  void save(Writer& w) const { w.vec_i32(w_); }
  void load(Reader& r);

 private:
  int n_, sets_;
  std::vector<int32_t> w_;  // sets_ x n_, 128 = weight 1.0, kept in 1..1024
};

}  // namespace cmix
