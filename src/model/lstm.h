// Level 2 specialist L: a small byte-level LSTM (as in cmix).
//
// It reads the data one byte at a time and keeps a running memory in its
// hidden state, so it can carry information further than the fixed-length
// contexts. Its output is a probability for each of the 256 next bytes; a
// prefix-sum tree turns that into P(next bit = 1) at any point in the byte.
//
// State split, like the rest of the model:
//   LstmState (in Stream) - hidden state and output of the current step:
//                           changes when the context moves (forward)
//   Lstm (in Model)       - weights, optimizer moments, training buffer:
//                           changes only when learning
// Training is online: the output layer every byte, and backpropagation
// through time over the last kHorizon bytes every kHorizon bytes (Adam).
#pragma once

#include <cstdint>
#include <vector>

#include "core/serial.h"
#include "model/stream_types.h"

namespace cmix {

class Lstm {
 public:
  static constexpr int kHorizon = 16;
  explicit Lstm(int cells);
  int cells() const { return h_; }

  // Moves the state forward over byte b (computes the next-byte distribution).
  void forward(LstmState& s, int b) const;
  // P(next bit = 1), 16-bit, from the current distribution.
  int p_bit(const LstmState& s, int bit_index, int partial) const;
  // Learns that byte b followed state s (call before forward(s, b)).
  void learn(const LstmState& s, int b);

  void save(Writer& w) const;
  void load(Reader& r);

 private:
  struct Param {  // weights with Adam moments
    std::vector<float> w, m, v, g;
    void init(size_t n, float scale, uint32_t& seed);
    void resize_like() { m.assign(w.size(), 0); v.assign(w.size(), 0); g.assign(w.size(), 0); }
  };
  struct Step {  // what backpropagation needs from one forward step
    int x = 0;
    std::vector<float> i, f, o, g, c_prev, h_prev, c, dh;
  };
  void adam(Param& p, float lr);
  void bptt();

  int h_;
  Param wx_;  // 4H x 256: input byte -> gates (one column per byte)
  Param wh_;  // 4H x H: previous hidden -> gates
  Param b_;   // 4H
  Param v_;   // 256 x H: hidden -> byte logits
  Param c_;   // 256
  std::vector<Step> ring_;
  uint32_t steps_ = 0;  // steps recorded since the last backpropagation
  uint64_t t_ = 0;      // Adam time step
};

}  // namespace cmix
