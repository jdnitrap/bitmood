// A Session walks one stream of bytes through a Model, bit by bit.
// It owns the position (Stream + BitPos); the Model owns what is learned.
#pragma once

#include <cstdint>

#include "model/model.h"

namespace cmix {

class Session {
 public:
  explicit Session(Model& m, const Stream& s = Stream()) : m_(m), s_(s) {}

  // P(next bit = 1) for the current bit. Must be called before learn_bit/skip_bit.
  int predict() { return m_.predict(s_, bp_, v_); }

  // Learn from the real bit, then move on. Returns the bit's cost in bits.
  double learn_bit(int bit);
  // Move on without learning (generation, seeding a loaded memory).
  void skip_bit(int bit);

  // Starts a new record (image, sound) at the current position.
  void begin_record() { m_.begin_record(s_); }

  // Whole-byte helpers.
  double learn_byte(uint8_t b);
  void feed_byte(uint8_t b);

  Model& model() { return m_; }
  const Model::Votes& votes() const { return v_; }
  const Stream& stream() const { return s_; }
  BitPos bitpos() const { return bp_; }

 private:
  void next(int bit);
  Model& m_;
  Stream s_;
  BitPos bp_;
  Model::Votes v_;
  double byte_bits_ = 0;  // cost of the byte in progress while learning
};

}  // namespace cmix
