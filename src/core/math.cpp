#include "core/math.h"

#include <cmath>

namespace cmix {

namespace {

// Stretch of the centre of each 1/4096-wide probability bucket.
struct StretchTable {
  float tab[4096];
  StretchTable() {
    for (int i = 0; i < 4096; ++i) {
      double p = (i + 0.5) / 4096.0;
      tab[i] = (float)std::log(p / (1.0 - p));
    }
  }
};

const StretchTable& stretch_table() {
  static const StretchTable t;
  return t;
}

}  // namespace

float stretch(int p) { return stretch_table().tab[clampi(p, kProbMin, kProbMax) >> 4]; }

float squash(float x) {
  x = clampf(x, -30.0f, 30.0f);
  return 1.0f / (1.0f + std::exp(-x));
}

int to_p16(float p) { return clampi((int)std::lround(p * (float)kProbOne), kProbMin, kProbMax); }

double bit_cost(int p, int bit) {
  double q = clampi(p, kProbMin, kProbMax) / (double)kProbOne;
  return -std::log2(bit ? q : 1.0 - q);
}

}  // namespace cmix
