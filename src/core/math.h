// Probability helpers shared by every module.
//
// Probabilities are 16-bit integers: p = P(bit = 1) * 65536, kept in 1..65535.
// "Stretch" space is the logit ln(p / (1 - p)) as a float; "squash" is its
// inverse. Mixing happens in stretch space.
#pragma once

#include <cstdint>

namespace cmix {

constexpr int kProbOne = 65536;  // P = 1.0
constexpr int kProbMin = 1;
constexpr int kProbMax = 65535;

inline int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// p (16-bit) -> logit, table-driven, in about -11..11.
float stretch(int p);
// logit -> probability in 0..1.
float squash(float x);
// Probability in 0..1 -> 16-bit p, clamped to 1..65535.
int to_p16(float p);

// Cost in bits of seeing `bit` when the model said P(1) = p / 65536.
double bit_cost(int p, int bit);

}  // namespace cmix
