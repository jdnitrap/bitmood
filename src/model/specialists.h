// The specialists. Each one looks at the context in its own way and
// returns P(next bit = 1) as 1..4094 (2048 = no opinion).
//
// predict() is const: it never changes what the specialist has learned.
// learn() is the only thing that does.
#pragma once

#include <cstdint>
#include <vector>

#include "core/serial.h"
#include "model/history.h"
#include "model/stream.h"

namespace cmix {

// Hash-indexed table of (n0, n1) bit counts. Pairs are halved when their
// sum passes `limit`, so recent evidence outweighs old evidence.
class CountTable {
 public:
  CountTable(int bits, int limit);
  int predict(uint32_t ctx) const;
  void learn(uint32_t ctx, int bit);
  void save(Writer& w) const;
  void load(Reader& r);

 private:
  uint32_t mask_;
  int limit_;
  std::vector<uint16_t> n0_, n1_;
};

// A: short recent context (last two bytes + bits so far).
class RecentPattern {
 public:
  RecentPattern() : table_(20, 4096) {}
  int predict(const History& h, const Stream& s, BitPos bp) const { return table_.predict(ctx(h, s, bp)); }
  void learn(const History& h, const Stream& s, BitPos bp, int bit) { table_.learn(ctx(h, s, bp), bit); }
  void save(Writer& w) const { table_.save(w); }
  void load(Reader& r) { table_.load(r); }

 private:
  static uint32_t ctx(const History& h, const Stream& s, BitPos bp);
  CountTable table_;
};

// B: find the longest earlier stretch that matches the most recent bytes
// and predict the bit that followed it. Scans up to 4096 bytes back.
class LongMatch {
 public:
  int predict(const History& h, BitPos bp) const;
};

// C: fixed prior from the MDBE byte-shape flags of the last byte.
class ByteShape {
 public:
  static bool is_alpha(int b) { return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z'); }
  static bool is_digit(int b) { return b >= '0' && b <= '9'; }
  static bool is_upper(int b) { return b >= 'A' && b <= 'Z'; }
  static bool is_punct(int b) {
    return (b >= 33 && b <= 47) || (b >= 58 && b <= 64) || (b >= 91 && b <= 96) || (b >= 123 && b <= 126);
  }
  static bool is_space(int b) { return b == ' ' || b == '\t' || b == '\n' || b == '\r'; }
  static bool utf8_lead(int b) { return (b & 0xC0) != 0x80 && b >= 0x80; }
  int predict(int last_byte, BitPos bp) const;
};

// D: order-N counts. Which N is decided by the hash the caller passes in.
class OrderN {
 public:
  OrderN() : table_(18, 2048) {}
  int predict(uint32_t key) const { return table_.predict(key); }
  void learn(uint32_t key, int bit) { table_.learn(key, bit); }
  void save(Writer& w) const { table_.save(w); }
  void load(Reader& r) { table_.load(r); }

 private:
  CountTable table_;
};

}  // namespace cmix
