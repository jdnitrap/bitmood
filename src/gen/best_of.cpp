#include "gen/best_of.h"

#include <cmath>

namespace cmix {

double score_candidate(const Candidate& c, const BestOfOptions& o) {
  double copy_excess = c.copy > o.copy_allowance ? (double)(c.copy - o.copy_allowance) : 0.0;
  return -std::fabs(c.bpb - o.target_bpb) - 0.05 * copy_excess - 0.25 * c.spikes;
}

Candidate best_of_line(Generator& g, const BestOfOptions& o, std::shared_ptr<const SuffixArray> training,
                       uint64_t seed, uint64_t unit) {
  const Generator::Snapshot start = g.snapshot();
  Candidate best;
  bool have = false;
  for (int k = 0; k < o.candidates; ++k) {
    g.restore(start);
    g.reseed(seed ^ (unit * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)k * 0xD1B54A32D192ED03ull));
    Candidate c;
    while (c.bytes.size() < o.max_len || g.must_continue()) {
      const double before = g.total_bits();
      const uint8_t b = g.next();
      c.bytes.push_back(b);
      c.bits.push_back(g.total_bits() - before);
      if (b == '\n' && !g.must_continue()) break;
    }
    double total = 0;
    for (double x : c.bits) {
      total += x;
      if (x > o.spike_bits) ++c.spikes;
    }
    c.bpb = c.bytes.empty() ? 0.0 : total / (double)c.bytes.size();
    if (training && !training->empty()) {
      CopyMeter meter(training);
      for (uint8_t b : c.bytes) meter.accept(b);
      c.copy = meter.longest();
    }
    c.score = score_candidate(c, o);
    if (!have || c.score > best.score) {
      best = c;
      have = true;
    }
  }
  // Replay the winner from the starting point.
  g.restore(start);
  for (size_t i = 0; i < best.bytes.size(); ++i) g.emit(best.bytes[i], best.bits[i]);
  return best;
}

}  // namespace cmix
