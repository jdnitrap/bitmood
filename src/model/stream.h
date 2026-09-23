// Where the model is in the data. Two small value types:
//
//   BitPos - which bit of the current byte is next, and the bits already decided.
//   Stream - byte-level context derived from finished bytes (hashes, last byte).
//
// Both are cheap to copy, so generation can snapshot and restore them.
// Nothing learned lives here; learned state lives in Model.
#pragma once

#include <cstdint>

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
};

struct Stream {
  int last_byte = -1;            // previous finished byte, -1 at start
  uint32_t order_hash[3] = {};   // D0/D1/D2 contexts: last 1/3/4 bytes
  uint64_t bytes = 0;            // bytes seen by this stream
};

}  // namespace cmix
