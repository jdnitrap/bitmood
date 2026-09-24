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

// The graph specialist's side of the context. Tokens are words (text),
// 20 ms slice shapes (audio) or 8-pixel run shapes (images).
struct GraphState {
  static constexpr uint32_t kNone = 0xFFFFFFFFu;
  static constexpr int kMaxWord = 31;
  uint32_t t0 = kNone, t1 = kNone;  // last two finished tokens, newest t0
  // Set by advance_byte when the byte just finished a token: the edge to learn.
  bool completed = false;
  uint32_t learn_t1 = kNone, learn_t0 = kNone, learn_next = kNone;
  // Generation: the token planned for the current unit (kNone = none).
  uint32_t plan = kNone;
  bool planned = false;             // planning already tried for this unit
  uint32_t expect = kNone;          // audio/image: the shape G expects now
  // Text: the word in progress (lowercase) and the last finished word.
  char word[kMaxWord + 1] = {};
  int wlen = 0;
  bool overflow = false;            // word too long to be a token
  bool capital = true;              // next word likely starts with a capital
  char done_word[kMaxWord + 1] = {};
  int done_len = 0;
  // Text: next-byte distribution from the graph, as a prefix-sum tree.
  float tree[512] = {};
  bool tree_ready = false;
  // Audio: the current slice so far (channel 0).
  uint32_t n = 0, zero_crossings = 0;
  int last_sign = 0;
  double sum_sq = 0, prev_rms = 0;
};

// The spiking network on the graph (SNN), per stream: which neurons are
// charged, which fired recently (for learning), and what it predicts.
struct SnnState {
  static constexpr uint32_t kNone = 0xFFFFFFFFu;
  static constexpr int kMaxCharged = 64;
  static constexpr int kMaxFired = 16;
  int n = 0;                          // charged neurons in use
  uint32_t tok[kMaxCharged] = {};     // neuron (graph node) ids
  float charge[kMaxCharged] = {};
  int nf = 0;                         // recently fired neurons
  uint32_t fired[kMaxFired] = {};
  float trace[kMaxFired] = {};        // eligibility: 1 when fired, decays each step
  uint32_t predicted = kNone;         // most-charged neuron after the last spike
  // Text: next-byte distribution from the charged words, as a prefix-sum tree.
  float tree[512] = {};
  bool tree_ready = false;
};

}  // namespace cmix
