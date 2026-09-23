// The model: everything that is learned and kept in a state file.
//
//   predict()      - const: combine the specialists' votes into one P(bit = 1)
//   learn()        - adjust counts, mixer weights and view scores from the real bit
//   advance_byte() - move the context forward over a finished byte
//
// Keeping learn() and advance_byte() separate is what lets generation walk
// forward through its own output without teaching itself that output.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/serial.h"
#include "model/config.h"
#include "model/grid.h"
#include "model/history.h"
#include "model/mixer.h"
#include "model/specialists.h"
#include "model/stream.h"

namespace cmix {

// Which grid view is predicting best right now. Each view's cost is a
// decayed average of bits spent by its best vote; "none" (index 0) is the
// cost of specialist A, the no-grid baseline. The winner selects the
// mixer's weight set, so each kind of region learns its own trust.
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
  static constexpr int kMaxInputs = 64;
  static constexpr int kBaseInputs = 6;  // A, B, C, D0, D1, D2
  struct Votes {
    int p[kMaxInputs] = {};
    int mixed = 2048;
    int set = 0;
  };

  explicit Model(const Config& cfg);

  const Config& config() const { return cfg_; }
  int num_inputs() const { return kBaseInputs + grid_.num_inputs(); }
  // Short code for input i: "A", "B", "C", "D0".."D2", "E0u", "E0r", ...
  std::string input_name(int i) const;
  // Longer description, e.g. "D1 order-3", "E line rows (above)", "E auto width 1 (row)".
  std::string input_label(int i) const;
  const Grid& grid() const { return grid_; }
  int best_view() const { return tracker_.best(); }

  int predict(const Stream& s, BitPos bp, Votes& v) const;
  void learn(const Stream& s, BitPos bp, const Votes& v, int bit);
  void advance_byte(Stream& s, uint8_t b);

  History& history() { return hist_; }
  const History& history() const { return hist_; }
  const Mixer& mixer() const { return mix_; }

  // Running totals over everything this model has learned from.
  uint64_t bytes_learned = 0;
  double bits_spent = 0;  // sum of -log2 P(real bit) while learning

  void save(Writer& w) const;
  void load(Reader& r);

 private:
  Config cfg_;
  History hist_;
  RecentPattern a_;
  LongMatch b_;
  ByteShape c_;
  OrderN d_[3];
  Grid grid_;
  ViewTracker tracker_;
  Mixer mix_;
};

// Order-N keys for D0/D1/D2 at a given bit position.
uint32_t order_key(const Stream& s, int which, BitPos bp);

}  // namespace cmix
