// Where the model is in the data. Two value types:
//
//   BitPos - which bit of the current byte is next, and the bits already decided.
//   Stream - byte-level context derived from finished bytes (order hashes,
//            words, byte classes, line positions, the current long match,
//            row-width statistics).
//
// Both are plain values, so generation can snapshot and restore them.
// Nothing learned lives here; learned state lives in Model.
#pragma once

#include <cstdint>

#include "model/width_finder.h"

namespace cmix {

struct BitPos {
  int index = 0;    // 0 = MSB .. 7 = LSB
  int partial = 0;  // bits already decided in this byte, MSB first
  // Moves to the next bit. Returns true when a whole byte is complete.
  bool push(int bit) {
    partial = (partial << 1) | bit;
    return ++index == 8;
  }
  int byte() const { return partial & 0xFF; }
  // index and partial in one number, 1..255 (a leading 1 marks the length).
  uint32_t key() const { return (1u << index) | (uint32_t)partial; }
};

struct Stream {
  static constexpr int kOrders = 8;
  static constexpr int kOrderLen[kOrders] = {0, 1, 2, 3, 4, 5, 6, 8};

  int last_byte = -1;               // previous finished byte, -1 at start
  uint64_t order_hash[kOrders] = {};  // hash of the last kOrderLen[i] bytes
  uint64_t bytes = 0;               // bytes seen by this stream
  // Words (letters only, lowercased): the one in progress and the one before.
  uint64_t word = 0, prev_word = 0;
  // Byte classes of the last 4 bytes, 4 bits each, newest lowest.
  uint32_t classes = 0;
  // Long match: absolute history position of the byte it predicts next.
  uint64_t match_ptr = 0;
  uint32_t match_len = 0;
  // Line rows (absolute history positions).
  uint64_t line_start = 0;       // first byte of the current line
  uint64_t prev_line_start = 0;  // first byte of the line above
  WidthFinder widths;
};

}  // namespace cmix
