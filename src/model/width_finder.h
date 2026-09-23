// Automatic row width: the "column count" slider of a hex editor, set by
// the data itself. Every distance d in 2..kMaxWidth is scored by how often
// a byte equals the byte d positions back (exponentially decayed, so the
// score follows the current region). The two best distances become the
// widths of the grid's Auto views.
#pragma once

#include <cstdint>

#include "core/serial.h"
#include "model/history.h"

namespace cmix {

struct WidthFinder {
  static constexpr int kMaxWidth = 1024;
  static constexpr int kPickEvery = 128;  // bytes between re-picks

  float score[kMaxWidth + 1] = {};
  int width[2] = {0, 0};  // 0 = no width stands out yet
  uint32_t since_pick = 0;

  void update(const History& h);  // call after each byte is pushed
  void save(Writer& w) const;
  void load(Reader& r);

 private:
  void pick();
};

}  // namespace cmix
