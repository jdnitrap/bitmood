// Small directly indexed adaptive tables.
//
//   AdaptiveMap - one adaptive probability per index (e.g. match length).
//   Apm         - "adaptive probability map" / SSE: refines a probability
//                 given a small context, by interpolating between 33 learned
//                 points along the stretch axis.
#pragma once

#include <cstdint>
#include <vector>

#include "core/serial.h"

namespace cmix {

class AdaptiveMap {
 public:
  AdaptiveMap(size_t n, int limit);
  int p(size_t i) const { return p_[i]; }
  void update(size_t i, int bit);
  void save(Writer& w) const;
  void load(Reader& r);

 private:
  int limit_;
  std::vector<uint16_t> p_, n_;
};

class Apm {
 public:
  explicit Apm(size_t contexts, int rate_bits = 7);
  // Refined probability (16-bit) for input probability p in context ctx.
  // `slot` records what update() needs.
  int refine(int p, size_t ctx, uint32_t& slot) const;
  void update(uint32_t slot, int bit);
  void save(Writer& w) const { w.u16_array(t_.data(), t_.size()); }
  void load(Reader& r) { r.u16_array(t_.data(), t_.size()); }

 private:
  size_t contexts_;
  int rate_;
  std::vector<uint16_t> t_;  // contexts x 33
};

}  // namespace cmix
