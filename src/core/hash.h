// Hash mixing for context keys.
#pragma once

#include <cstdint>

namespace cmix {

// Fold v into h. Same function the original prototype used for specialist A.
inline uint32_t hash_mix(uint32_t h, uint32_t v) {
  h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
  return h * 0x85ebca6bu;
}

}  // namespace cmix
