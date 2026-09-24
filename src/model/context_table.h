// One shared hash table for every context-based specialist.
//
// Each slot holds an adaptive probability for "next bit = 1" in one context:
//   check - 16 bits of the context's hash, to detect collisions
//   p     - P(1) * 65536
//   n     - how many times the slot was updated (sets the learning rate)
// Slots come in buckets of 4. A context looks only in its own bucket; if it
// is not there, learn() claims an empty slot or the least-used one.
//
// lookup() is const and never claims, so predictions don't change the table.
#pragma once

#include <cstdint>
#include <vector>

#include "core/serial.h"

namespace cmix {

class ContextTable {
 public:
  static constexpr int kBucket = 4;
  explicit ContextTable(int bits);  // 2^bits slots, 8 bytes each

  // Slot index for hash h, or -1 if the context has not been seen.
  int64_t lookup(uint64_t h) const;
  // Slot index for hash h, claiming one if needed.
  uint32_t claim(uint64_t h);

  int p(uint32_t i) const { return slots_[i].p; }
  int n(uint32_t i) const { return slots_[i].n; }
  // Moves p toward the bit at rate 1/(n + 1.5). n stops growing at `limit`.
  // A new region halves n (halve_counts) so a familiar context can move again.
  void update(uint32_t i, int bit, int limit);
  // First miss of h is remembered and returns false. The next miss of the
  // same check returns true, and the caller may claim a slot. One pending
  // check per bucket: a different context replaces it.
  bool promote(uint64_t h);
  // Halve every slot's update count. Used when a new region starts.
  void halve_counts();

  size_t size() const { return slots_.size(); }
  void save(Writer& w) const;
  void load(Reader& r);

 private:
  struct Slot {
    uint16_t check = 0;  // 0 = empty
    uint16_t p = 32768;
    uint16_t n = 0;
    uint16_t pad = 0;
  };
  static uint16_t check_of(uint64_t h) { return (uint16_t)((h & 0xFFFF) | 1); }
  size_t bucket_of(uint64_t h) const { return (size_t)((h >> 16) & mask_) * kBucket; }
  std::vector<Slot> slots_;
  std::vector<uint16_t> witness_;  // one pending check per bucket, 0 = none
  uint64_t mask_;                  // number of buckets - 1
};

}  // namespace cmix
