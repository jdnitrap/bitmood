#include "model/model.h"

#include <algorithm>
#include <stdexcept>

#include "core/hash.h"
#include "core/math.h"

namespace cmix {

namespace {

// Hash of the last n bytes of history, so order-N contexts recur.
uint32_t hash_last(const History& h, size_t n, uint32_t seed) {
  uint32_t x = seed;
  for (size_t i = 0; i < n && i < h.size(); ++i) x = hash_mix(x, h.back(i));
  return x;
}

std::vector<int> initial_weights(int grid_inputs) {
  std::vector<int> w(Model::kBaseInputs, 128);
  w.insert(w.end(), grid_inputs, 32);  // new views start with a small say
  return w;
}

}  // namespace

// ---- ViewTracker ------------------------------------------------------------

void ViewTracker::learn(int baseline_p, const int* grid_p, int bit) {
  static constexpr float kRate = 1.0f / 1024.0f;  // ~128 bytes of memory
  auto track = [&](float& c, int p) { c += kRate * ((float)bit_cost(p, bit) - c); };
  track(cost_[0], baseline_p);
  for (size_t i = 0; i < input_cost_.size(); ++i) track(input_cost_[i], grid_p[i]);
  best_ = 0;
  for (size_t v = 1; v < cost_.size(); ++v) {
    cost_[v] = std::min(input_cost_[2 * (v - 1)], input_cost_[2 * (v - 1) + 1]);
    if (cost_[v] < cost_[best_]) best_ = (int)v;
  }
}

void ViewTracker::save(Writer& w) const {
  w.vec_f32(cost_);
  w.vec_f32(input_cost_);
  w.i32(best_);
}

void ViewTracker::load(Reader& r) {
  size_t a = cost_.size(), b = input_cost_.size();
  r.vec_f32(cost_);
  r.vec_f32(input_cost_);
  if (cost_.size() != a || input_cost_.size() != b) throw std::runtime_error("state file: view count mismatch");
  best_ = r.i32();
}

// ---- Model ------------------------------------------------------------------

uint32_t order_key(const Stream& s, int which, BitPos bp) {
  const uint32_t k = (uint32_t)bp.index, part = (uint32_t)bp.partial;
  switch (which) {
    case 0: return s.order_hash[0] ^ k * 0x9e3779b9u ^ part;
    case 1: return s.order_hash[1] ^ (k + 1) * 0x85ebca6bu ^ part << 3;
    default: return s.order_hash[2] ^ (k << 16) ^ part;
  }
}

Model::Model(const Config& cfg)
    : cfg_(cfg),
      grid_(cfg.views()),
      tracker_(grid_.num_views()),
      mix_(initial_weights(grid_.num_inputs()), grid_.num_views() + 1) {}

std::string Model::input_name(int i) const {
  static const char* base[kBaseInputs] = {"A", "B", "C", "D0", "D1", "D2"};
  if (i < kBaseInputs) return base[i];
  int g = i - kBaseInputs;
  return "E" + std::to_string(g / 2) + (g % 2 ? "r" : "u");
}

std::string Model::input_label(int i) const {
  static const char* base[kBaseInputs] = {"A recent pattern", "B long match",   "C byte shape",
                                          "D0 order-1",       "D1 order-3",     "D2 order-4"};
  if (i < kBaseInputs) return base[i];
  int g = i - kBaseInputs;
  return "E " + view_name(grid_.view(g / 2)) + (g % 2 ? " (row)" : " (above)");
}

int Model::predict(const Stream& s, BitPos bp, Votes& v) const {
  v.p[0] = a_.predict(hist_, s, bp);
  v.p[1] = b_.predict(hist_, bp);
  v.p[2] = c_.predict(s.last_byte, bp);
  for (int i = 0; i < 3; ++i) v.p[3 + i] = d_[i].predict(order_key(s, i, bp));
  grid_.predict(hist_, s, bp, v.p + kBaseInputs);
  v.set = tracker_.best();
  v.mixed = mix_.mix(v.p, v.set);
  return v.mixed;
}

void Model::learn(const Stream& s, BitPos bp, const Votes& v, int bit) {
  mix_.learn(v.p, bit, v.mixed, v.set);
  a_.learn(hist_, s, bp, bit);
  for (int i = 0; i < 3; ++i) d_[i].learn(order_key(s, i, bp), bit);
  grid_.learn(hist_, s, bp, bit);
  tracker_.learn(v.p[0], v.p + kBaseInputs, bit);
}

void Model::advance_byte(Stream& s, uint8_t b) {
  hist_.push(b);
  s.last_byte = b;
  ++s.bytes;
  if (b == '\n') {
    s.prev_line_start = s.line_start;
    s.line_start = hist_.end();
  }
  s.widths.update(hist_);
  // D0/D1/D2 are order-1/3/4: hash only the last N bytes so contexts recur.
  s.order_hash[0] = hash_last(hist_, 1, 0x1000193u);
  s.order_hash[1] = hash_last(hist_, 3, 0x3000193u);
  s.order_hash[2] = hash_last(hist_, 4, 0x4000193u);
}

void Model::save(Writer& w) const {
  w.tag("HIST");
  w.u64(hist_.base());
  w.vec_u8(hist_.data());
  w.tag("SPEC");
  a_.save(w);
  for (const auto& d : d_) d.save(w);
  w.tag("GRID");
  grid_.save(w);
  tracker_.save(w);
  w.tag("MIXR");
  mix_.save(w);
  w.tag("STAT");
  w.u64(bytes_learned);
  w.f64(bits_spent);
}

void Model::load(Reader& r) {
  r.expect_tag("HIST");
  uint64_t base = r.u64();
  std::vector<uint8_t> bytes;
  r.vec_u8(bytes);
  hist_.assign(std::move(bytes), base);
  r.expect_tag("SPEC");
  a_.load(r);
  for (auto& d : d_) d.load(r);
  r.expect_tag("GRID");
  grid_.load(r);
  tracker_.load(r);
  r.expect_tag("MIXR");
  mix_.load(r);
  r.expect_tag("STAT");
  bytes_learned = r.u64();
  bits_spent = r.f64();
}

}  // namespace cmix
