#include "model/model.h"

#include "core/hash.h"

namespace cmix {

namespace {

// Hash of the last n bytes of history, so order-N contexts recur.
uint32_t hash_last(const History& h, size_t n, uint32_t seed) {
  uint32_t x = seed;
  for (size_t i = 0; i < n && i < h.size(); ++i) x = hash_mix(x, h.back(i));
  return x;
}

}  // namespace

uint32_t order_key(const Stream& s, int which, BitPos bp) {
  const uint32_t k = (uint32_t)bp.index, part = (uint32_t)bp.partial;
  switch (which) {
    case 0: return s.order_hash[0] ^ k * 0x9e3779b9u ^ part;
    case 1: return s.order_hash[1] ^ (k + 1) * 0x85ebca6bu ^ part << 3;
    default: return s.order_hash[2] ^ (k << 16) ^ part;
  }
}

Model::Model(const Config& cfg) : cfg_(cfg), mix_(kInputs) {}

const char* Model::input_name(int i) {
  static const char* names[kInputs] = {"A", "B", "C", "D0", "D1", "D2"};
  return (i >= 0 && i < kInputs) ? names[i] : "?";
}

int Model::predict(const Stream& s, BitPos bp, Votes& v) const {
  v.p[0] = a_.predict(hist_, s, bp);
  v.p[1] = b_.predict(hist_, bp);
  v.p[2] = c_.predict(s.last_byte, bp);
  for (int i = 0; i < 3; ++i) v.p[3 + i] = d_[i].predict(order_key(s, i, bp));
  v.mixed = mix_.mix(v.p);
  return v.mixed;
}

void Model::learn(const Stream& s, BitPos bp, const Votes& v, int bit) {
  mix_.learn(v.p, bit, v.mixed);
  a_.learn(hist_, s, bp, bit);
  for (int i = 0; i < 3; ++i) d_[i].learn(order_key(s, i, bp), bit);
}

void Model::advance_byte(Stream& s, uint8_t b) {
  hist_.push(b);
  s.last_byte = b;
  ++s.bytes;
  // D0/D1/D2 are order-1/3/4: hash only the last N bytes so contexts recur.
  s.order_hash[0] = hash_last(hist_, 1, 0x1000193u);
  s.order_hash[1] = hash_last(hist_, 3, 0x3000193u);
  s.order_hash[2] = hash_last(hist_, 4, 0x4000193u);
}

void Model::save(Writer& w) const {
  w.tag("HIST");
  w.vec_u8(hist_.data());
  w.tag("SPEC");
  a_.save(w);
  for (const auto& d : d_) d.save(w);
  w.tag("MIXR");
  mix_.save(w);
  w.tag("STAT");
  w.u64(bytes_learned);
  w.f64(bits_spent);
}

void Model::load(Reader& r) {
  r.expect_tag("HIST");
  r.vec_u8(hist_.mutable_data());
  r.expect_tag("SPEC");
  a_.load(r);
  for (auto& d : d_) d.load(r);
  r.expect_tag("MIXR");
  mix_.load(r);
  r.expect_tag("STAT");
  bytes_learned = r.u64();
  bits_spent = r.f64();
}

}  // namespace cmix
