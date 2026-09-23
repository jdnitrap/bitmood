#include "model/session.h"

#include "core/math.h"

namespace cmix {

void Session::next(int bit) {
  if (bp_.push(bit)) {
    m_.advance_byte(s_, (uint8_t)bp_.byte());
    bp_ = BitPos();
  }
}

double Session::learn_bit(int bit) {
  double cost = bit_cost(v_.mixed, bit);
  m_.learn(s_, bp_, v_, bit);
  m_.bits_spent += cost;
  byte_bits_ += cost;
  if (bp_.index == 7) {
    m_.learn_byte(s_, (uint8_t)(((bp_.partial << 1) | bit) & 0xFF));
    m_.note_byte_cost(byte_bits_);
    ++m_.bytes_learned;
    byte_bits_ = 0;
  }
  next(bit);
  return cost;
}

void Session::skip_bit(int bit) { next(bit); }

double Session::learn_byte(uint8_t b) {
  double cost = 0;
  for (int k = 7; k >= 0; --k) {
    predict();
    cost += learn_bit((b >> k) & 1);
  }
  return cost;
}

void Session::feed_byte(uint8_t b) {
  for (int k = 7; k >= 0; --k) skip_bit((b >> k) & 1);
}

}  // namespace cmix
