// Hash mixing for context keys.
#pragma once

#include <cstdint>

namespace cmix {

// Fold v into a 32-bit h (used for byte-level running hashes).
inline uint32_t hash_mix(uint32_t h, uint32_t v) {
  h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
  return h * 0x85ebca6bu;
}

// 64-bit finalizer (splitmix64): spreads every input bit over the result.
inline uint64_t hash_fin(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}

// Combine a running 64-bit hash with one more value.
inline uint64_t hash_add(uint64_t h, uint64_t v) { return hash_fin(h ^ (v + 0x9e3779b97f4a7c15ull + (h << 6))); }

}  // namespace cmix
