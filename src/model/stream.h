// Where the model is in the data. Two value types:
//
//   BitPos - which bit of the current byte is next, and the bits already decided.
//   Stream - byte-level context derived from finished bytes (hashes, line
//            positions, row-width statistics).
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
  int last_byte = -1;            // previous finished byte, -1 at start
  uint32_t order_hash[3] = {};   // D0/D1/D2 contexts: last 1/3/4 bytes
  uint64_t bytes = 0;            // bytes seen by this stream
  // Line rows (absolute history positions).
  uint64_t line_start = 0;       // first byte of the current line
  uint64_t prev_line_start = 0;  // first byte of the line above
  WidthFinder widths;
};

}  // namespace cmix
