#include "core/math.h"

#include <cmath>

namespace cmix {

namespace {

struct StretchTable {
  int tab[4096];
  StretchTable() {
    for (int i = 1; i < 4095; ++i) {
      double pr = i / 4095.0;
      double s = std::log(pr / (1.0 - pr));
      tab[i] = clampi((int)std::lround(s * 256.0), -2047, 2047);
    }
    tab[0] = -2047;
    tab[4095] = 2047;
  }
};

const StretchTable& stretch_table() {
  static const StretchTable t;
  return t;
}

}  // namespace

int stretch(int p) { return stretch_table().tab[clampi(p, 1, 4094)]; }

int squash(int z) {
  double x = z / 256.0;
  if (x > 20) return 4094;
  if (x < -20) return 1;
  double p = 1.0 / (1.0 + std::exp(-x));
  return clampi((int)std::lround(p * 4095.0), 1, 4094);
}

double bit_cost(int p, int bit) {
  double q = clampi(p, 1, 4094) / 4096.0;
  return -std::log2(bit ? q : 1.0 - q);
}

}  // namespace cmix
