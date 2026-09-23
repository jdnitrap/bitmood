#include "model/grid.h"

#include <algorithm>

#include "core/hash.h"

namespace cmix {

// ---- WidthFinder ------------------------------------------------------------

void WidthFinder::update(const History& h) {
  static constexpr float kDecay = 1.0f - 1.0f / 2048.0f;
  const size_t n = h.size();
  if (n < 3) return;
  const uint8_t now = h.back(0);
  const int top = (int)std::min<size_t>(kMaxWidth, n - 1);
  for (int d = 2; d <= top; ++d) score[d] = score[d] * kDecay + (h.back((size_t)d) == now ? 1.0f : 0.0f);
  if (++since_pick >= kPickEvery) {
    since_pick = 0;
    pick();
  }
}

void WidthFinder::pick() {
  double mean = 0;
  for (int d = 2; d <= kMaxWidth; ++d) mean += score[d];
  mean /= (kMaxWidth - 1);
  // A width must stand out from the average distance, or there is no grid.
  auto stands_out = [&](int d) { return score[d] >= 4.0f && score[d] >= 1.25 * mean; };
  auto related = [](int a, int b) { return a % b == 0 || b % a == 0; };
  // Multiples of a record size score about as well as the record size itself,
  // so take the smallest width within 3% of the top score (the "fundamental").
  auto smallest_near_top = [&](int skip_related_to) {
    float top = 0.0f;
    for (int d = 2; d <= kMaxWidth; ++d)
      if (!(skip_related_to && related(d, skip_related_to))) top = std::max(top, score[d]);
    if (top <= 0.0f) return 0;
    for (int d = 2; d <= kMaxWidth; ++d)
      if (!(skip_related_to && related(d, skip_related_to)) && score[d] >= 0.97f * top) return d;
    return 0;
  };
  const int best = smallest_near_top(0);
  const int second = best ? smallest_near_top(best) : 0;
  width[0] = (best && stands_out(best)) ? best : 0;
  width[1] = (second && stands_out(second)) ? second : 0;
}

void WidthFinder::save(Writer& w) const {
  for (float s : score) w.f32(s);
  w.i32(width[0]);
  w.i32(width[1]);
  w.u32(since_pick);
}

void WidthFinder::load(Reader& r) {
  for (float& s : score) s = r.f32();
  width[0] = r.i32();
  width[1] = r.i32();
  since_pick = r.u32();
}

// ---- views ------------------------------------------------------------------

std::string view_label(const View& v, const Stream& s) {
  switch (v.kind) {
    case ViewKind::Line: return "line rows";
    case ViewKind::Auto: {
      int w = s.widths.width[v.param];
      return w ? "width " + std::to_string(w) : "no width yet";
    }
    case ViewKind::Word: return "word " + std::to_string(v.param);
  }
  return "?";
}

std::string view_name(const View& v) {
  if (v.kind == ViewKind::Auto) return "auto width " + std::to_string(v.param + 1);
  return view_label(v, Stream());
}

GridCell grid_cell(const View& v, const History& h, const Stream& s) {
  GridCell c;
  const uint64_t pos = h.end();  // absolute position of the byte being predicted
  auto byte_at = [&](uint64_t q, uint64_t limit) { return (q < limit && h.has(q)) ? (int)h.at(q) : 256; };
  if (v.kind == ViewKind::Line) {
    c.valid = true;
    c.col = (int)std::min<uint64_t>(pos - s.line_start, 255);
    const uint64_t q = s.prev_line_start + (pos - s.line_start);
    c.above = byte_at(q, s.line_start);
    c.above_right = byte_at(q + 1, s.line_start);
    return c;
  }
  const int w = v.kind == ViewKind::Auto ? s.widths.width[v.param] : v.param;
  if (w <= 0 || pos < (uint64_t)w) return c;
  c.valid = true;
  c.width = w;
  c.col = (int)(pos % (uint64_t)w);
  c.above = byte_at(pos - w, pos);
  c.above_right = byte_at(pos - w + 1, pos);
  return c;
}

// ---- Grid specialist --------------------------------------------------------

void Grid::contexts(const History& h, const Stream& s, uint64_t* out) const {
  for (int i = 0; i < num_views(); ++i) {
    const GridCell c = grid_cell(views_[i], h, s);
    const uint64_t id = (uint64_t)i + (c.valid ? 0 : 0x80);
    out[2 * i] = hash_add(hash_add(0xE1000000ull + id, (uint64_t)c.above), (uint64_t)c.above_right);
    out[2 * i + 1] =
        hash_add(hash_add(hash_add(0xE2000000ull + id, (uint64_t)c.above), (uint64_t)(s.last_byte + 1)), (uint64_t)c.col);
  }
}

}  // namespace cmix
