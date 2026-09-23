// Specialist B, long match: find the most recent earlier place where the
// last kMinLen bytes occurred, and predict that what followed then follows
// now. A hash index (last kMinLen bytes -> position after them) finds the
// place in one lookup; once found, the match is followed byte by byte
// until it breaks.
//
// The index belongs to the model (it covers the history); the current
// match (pointer and length) belongs to the Stream.
#pragma once

#include <cstdint>
#include <vector>

#include "core/serial.h"
#include "model/adaptive.h"
#include "model/history.h"
#include "model/stream.h"

namespace cmix {

class MatchModel {
 public:
  static constexpr int kMinLen = 5;
  static constexpr int kInputs = 2;
  explicit MatchModel(int index_bits = 22);

  // Call after each byte is pushed to history. Updates s.match_*.
  void advance(const History& h, Stream& s);
  // Rebuilds the index from history (after loading or after old history is dropped).
  void rebuild(const History& h);

  // The byte the match expects next, or -1 if none / it disagrees with the
  // bits already decided in this byte.
  static int expected(const History& h, const Stream& s, BitPos bp);
  // Mixer selector: 0 no match, 1 short, 2 medium, 3 long.
  static int state(const History& h, const Stream& s, BitPos bp);

  // Two votes in stretch space; slot records what learn() needs.
  void predict(const History& h, const Stream& s, BitPos bp, float* x, int* p, int& slot) const;
  void learn(int slot, int bit);

  void save(Writer& w) const { map_.save(w); }
  void load(Reader& r) { map_.load(r); }

 private:
  static int length_bucket(uint32_t len);
  uint32_t key(const History& h) const;  // index slot for the last kMinLen bytes
  int bits_;
  std::vector<uint32_t> index_;  // history index + 1 (0 = empty)
  AdaptiveMap map_;              // (length bucket, expected bit) -> P(bit = 1)
};

}  // namespace cmix
