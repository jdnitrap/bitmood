// The grid: the file laid out in rows and columns, the way a hex editor
// shows it. Each *view* is one way of cutting rows:
//
//   Line  - a row ends at each newline (text, code, CSS, tables)
//   Auto  - fixed row width, chosen automatically from the data (the
//           "column count" slider, found by measuring which distance
//           between equal bytes is most common)
//   Word  - fixed width of 2, 4 or 8 bytes (16/32/64-bit word views)
//
// For the byte being predicted, a view gives its column and the byte
// directly above it (same column, previous row). The Grid specialist turns
// that into votes; WidthFinder keeps the statistics behind the Auto views.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/serial.h"
#include "model/history.h"
#include "model/stream.h"
#include "model/width_finder.h"

namespace cmix {

enum class ViewKind : uint8_t { Line = 0, Auto = 1, Word = 2 };

struct View {
  ViewKind kind;
  int param;  // Auto: slot 0/1 in WidthFinder; Word: width in bytes
};

// Where the current byte sits in a view's grid.
struct GridCell {
  bool valid = false;  // false if the view has no row width yet
  int above = 256;     // byte above, 256 = nothing there
  int above_right = 256;
  int col = 0;
  int width = 0;       // row width (0 for line rows, which vary)
};

// Describes view `v` in words, e.g. "line rows", "width 24", "word 4".
std::string view_label(const View& v, const Stream& s);
// Like view_label but stable over time: "line rows", "auto width 1", "word 4".
std::string view_name(const View& v);

GridCell grid_cell(const View& v, const History& h, const Stream& s);

// The grid specialist: two votes per view.
//   vote 1: (byte above, byte above-right)       - vertical pattern
//   vote 2: (byte above, byte to the left, col)   - pattern within the row
// It only describes the contexts; their statistics live in the model's
// shared ContextTable like every other context specialist.
class Grid {
 public:
  explicit Grid(std::vector<View> views) : views_(std::move(views)) {}
  int num_views() const { return (int)views_.size(); }
  int num_inputs() const { return 2 * num_views(); }
  const View& view(int i) const { return views_[i]; }

  // Byte-level context hashes, two per view (the caller adds the bit position).
  void contexts(const History& h, const Stream& s, uint64_t* out) const;

 private:
  std::vector<View> views_;
};

}  // namespace cmix
