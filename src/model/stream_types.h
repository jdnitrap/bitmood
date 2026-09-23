// Small per-stream state structs used by Stream (kept separate so model
// components can use them without including all of Stream).
#pragma once

#include <cstdint>

namespace cmix {

// The LSTM's side of the context: hidden state and the current step's
// activations (for learning), plus the next-byte distribution as a
// prefix-sum tree: tree[1] = 1, children of node n are 2n and 2n+1,
// leaves 256..511 are the byte probabilities.
struct LstmState {
  static constexpr int kMaxCells = 64;
  float h[kMaxCells] = {}, c[kMaxCells] = {};
  float h_prev[kMaxCells] = {}, c_prev[kMaxCells] = {};
  float gi[kMaxCells] = {}, gf[kMaxCells] = {}, go[kMaxCells] = {}, gg[kMaxCells] = {};
  int x = -1;          // byte that produced this step (-1: no step yet)
  float tree[512] = {};
  bool ready = false;  // false until the first forward step
};

}  // namespace cmix
