// The model: everything that is learned and kept in a state file.
//
//   predict()      - const: every specialist votes, the mixers combine the
//                    votes, two APM stages refine the result
//   learn()        - update every table, mixer and APM from the real bit
//   advance_byte() - move the context forward over a finished byte
//
// Keeping learn() and advance_byte() separate is what lets generation walk
// forward through its own output without teaching itself that output.
//
// Specialists (inputs to the mixers):
//   O0, D1, A (order 2), D3, D4, D5, D6, D8  order-N contexts
//   W1, W2                                   words (text)
//   C1, C2                                   byte classes (learned MDBE flags)
//   I1..I6                                   image: neighbouring pixels (image)
//   S1..S4                                   audio: earlier samples (audio)
//   E..                                      grid views, two votes each
//   B1, B2                                   long match
//   bias
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "core/serial.h"
#include "model/adaptive.h"
#include "model/config.h"
#include "model/context_table.h"
#include "model/grid.h"
#include "model/history.h"
#include "model/match_model.h"
#include "model/mixer.h"
#include "model/stream.h"

namespace cmix {

// Which grid view is predicting best right now. Each view's cost is a
// decayed average of bits spent by its best vote; "none" (index 0) is the
// cost of the order-2 context, the no-grid baseline. The winner selects a
// mixer weight set, so each kind of region learns its own trust.
class ViewTracker {
 public:
  explicit ViewTracker(int views) : cost_(views + 1, 1.0f), input_cost_(2 * views, 1.0f) {}
  int best() const { return best_; }  // 0 = none, 1..views = view index + 1
  void learn(int baseline_p, const int* grid_p, int bit);
  void save(Writer& w) const;
  void load(Reader& r);

 private:
  std::vector<float> cost_, input_cost_;
  int best_ = 0;
};

class Model {
 public:
  static constexpr int kMaxInputs = 48;
  static constexpr int kMixers = 3;

  // Everything predict() works out that learn() needs, plus per-input
  // probabilities for reports (demo, compare).
  struct Votes {
    int n = 0;                     // number of inputs
    float x[kMaxInputs] = {};      // votes in stretch space
    int p[kMaxInputs] = {};        // each vote as a probability (16-bit)
    uint64_t ctx[kMaxInputs] = {}; // context hash for table-backed inputs
    int match_slot = -1;
    int sel[kMixers] = {};         // weight set per mixer
    float z[kMixers] = {};         // mixer outputs, stretch space
    int final_set = 0;
    float zf = 0;                  // final mixer output
    uint32_t apm_slot[2] = {};
    int mixed = 32768;             // final P(bit = 1), 16-bit
  };

  explicit Model(const Config& cfg);

  const Config& config() const { return cfg_; }
  int num_inputs() const { return n_inputs_; }
  std::string input_name(int i) const;   // short: "A", "D3", "W1", "E0u", "B1", ...
  std::string input_label(int i) const;  // longer, for reports
  const Grid& grid() const { return grid_; }
  int best_view() const { return tracker_.best(); }

  int predict(const Stream& s, BitPos bp, Votes& v) const;
  void learn(const Stream& s, BitPos bp, const Votes& v, int bit);
  void advance_byte(Stream& s, uint8_t b);
  // Marks the start of a new record (an image or a sound) at the current
  // position, so pixel and sample positions count from here.
  void begin_record(Stream& s) const { s.record_start = hist_.end(); }

  History& history() { return hist_; }
  const History& history() const { return hist_; }
  const Mixer& mixer(int k) const { return mix_[(size_t)k]; }
  const Mixer& final_mixer() const { return final_; }

  // Running totals over everything this model has learned from.
  uint64_t bytes_learned = 0;
  double bits_spent = 0;  // sum of -log2 P(real bit) while learning
  double recent_bpb = 0;  // bits/byte over roughly the last 4096 bytes learned
  void note_byte_cost(double bits) {
    const double rate = std::max(1.0 / 4096.0, 1.0 / (double)(bytes_learned + 1));
    recent_bpb += rate * (bits - recent_bpb);
  }

  void save(Writer& w) const;
  void load(Reader& r);

 private:
  void contexts(const Stream& s, uint64_t* out) const;  // byte-level hashes of table inputs
  void image_contexts(const Stream& s, uint64_t* out) const;
  void audio_contexts(const Stream& s, uint64_t* out) const;
  int typed_first_ = 0;  // first image/audio input

  Config cfg_;
  History hist_;
  Grid grid_;
  // Input layout (indices into Votes arrays).
  int n_ctx_ = 0;       // table-backed inputs come first
  int grid_first_ = 0;  // first grid input
  int match_first_ = 0;
  int bias_ = 0;
  int n_inputs_ = 0;
  ContextTable table_;
  MatchModel match_;
  ViewTracker tracker_;
  std::vector<Mixer> mix_;
  Mixer final_;
  Apm apm1_, apm2_;
};

}  // namespace cmix
