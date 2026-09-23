// Probability helpers shared by every module.
// Probabilities are 12-bit integers: 0..4095 means P(bit = 1) * 4096.
// "Stretch" space is ln(p / (1 - p)) scaled by 256, clamped to +-2047.
#pragma once

#include <cstdint>

namespace cmix {

inline int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }

// p (1..4094) -> stretch(p) in -2047..2047. Table-driven.
int stretch(int p);

// stretch value -> p in 1..4094.
int squash(int z);

// Cost in bits of seeing `bit` when the model said P(1) = p/4096.
double bit_cost(int p, int bit);

}  // namespace cmix
