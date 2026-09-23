// The model: everything that is learned and kept in a state file.
//
//   predict()      - const: combine the specialists' votes into one P(bit = 1)
//   learn()        - adjust counts and mixer weights from the real bit
//   advance_byte() - move the context forward over a finished byte
//
// Keeping learn() and advance_byte() separate is what lets generation walk
// forward through its own output without teaching itself that output.
#pragma once

#include <cstdint>

#include "core/serial.h"
#include "model/config.h"
#include "model/history.h"
#include "model/mixer.h"
#include "model/specialists.h"
#include "model/stream.h"

namespace cmix {

class Model {
 public:
  static constexpr int kInputs = 6;
  struct Votes {
    int p[kInputs] = {};
    int mixed = 2048;
  };

  explicit Model(const Config& cfg);

  const Config& config() const { return cfg_; }
  static const char* input_name(int i);  // "A", "B", "C", "D0", ...

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
  Mixer mix_;
};

// Order-N keys for D0/D1/D2 at a given bit position.
uint32_t order_key(const Stream& s, int which, BitPos bp);

}  // namespace cmix
