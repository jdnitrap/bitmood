#include "gen/best_of.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cmix {

double score_candidate(const Candidate& c, const BestOfOptions& o) {
  double copy_excess = c.copy > o.copy_allowance ? (double)(c.copy - o.copy_allowance) : 0.0;
  double s = -std::fabs(c.bpb - o.target_bpb) - 0.05 * copy_excess - 0.25 * c.spikes;
  if (o.typical_len > 0) s -= std::fabs((double)c.bytes.size() - o.typical_len) / o.typical_len;
  return s;
}

double typical_line_length(const std::vector<uint8_t>& text) {
  size_t lines = 1;
  for (uint8_t b : text) lines += b == '\n';
  const double len = (double)text.size() / (double)lines;
  return len < 10 ? 10 : (len > 200 ? 200 : len);
}

Candidate best_of_line(Generator& g, const BestOfOptions& o, std::shared_ptr<const SuffixArray> training,
                       uint64_t seed, uint64_t unit) {
  const Generator::Snapshot start = g.snapshot();
  const double satisfied_before = g.satisfaction();
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
    c.score = score_candidate(c, o) + o.goal_weight * (g.satisfaction() - satisfied_before);
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

namespace {

void slice_tone(const uint8_t* p, size_t n, int channels, double& rms, double& zc) {
  const int frame = std::max(2, 2 * channels);
  double sum = 0;
  int count = 0, crosses = 0, prev = 0;
  for (size_t i = 0; i + 1 < n; i += (size_t)frame) {
    const int16_t s = (int16_t)(p[i] | (p[i + 1] << 8));
    sum += (double)s * (double)s;
    const int sign = s > 0 ? 1 : (s < 0 ? -1 : 0);
    if (sign != 0 && prev != 0 && sign != prev) ++crosses;
    if (sign != 0) prev = sign;
    ++count;
  }
  rms = count > 0 ? std::sqrt(sum / (double)count) : 0;
  zc = count > 0 ? (double)crosses / (double)count : 0;
}

}  // namespace

void best_of_audio(Generator& g, int candidates, int slice_bytes, int channels, double target_rms,
                   double target_zc, uint64_t seed, long long nbytes) {
  if (candidates < 1) candidates = 1;
  if (slice_bytes < 2) slice_bytes = 2;
  const double rms_scale = std::max(target_rms, 1.0);
  uint64_t unit = 0;
  while ((long long)g.output().size() < nbytes) {
    const int want = (int)std::min<long long>(slice_bytes, nbytes - (long long)g.output().size());
    const Generator::Snapshot start = g.snapshot();
    std::vector<uint8_t> best_bytes;
    std::vector<double> best_bits;
    double best_score = 0;
    bool have = false;
    for (int k = 0; k < candidates; ++k) {
      g.restore(start);
      g.reseed(seed ^ (unit * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)k * 0xD1B54A32D192ED03ull));
      std::vector<uint8_t> bytes;
      std::vector<double> bits;
      bytes.reserve((size_t)want);
      for (int i = 0; i < want; ++i) {
        const double before = g.total_bits();
        bytes.push_back(g.next());
        bits.push_back(g.total_bits() - before);
      }
      double rms = 0, zc = 0;
      slice_tone(bytes.data(), bytes.size(), channels, rms, zc);
      const double score = -std::fabs(rms - target_rms) / rms_scale - std::fabs(zc - target_zc);
      if (!have || score > best_score) {
        best_bytes = std::move(bytes);
        best_bits = std::move(bits);
        best_score = score;
        have = true;
      }
    }
    g.restore(start);
    for (size_t i = 0; i < best_bytes.size(); ++i) g.emit(best_bytes[i], best_bits[i]);
    ++unit;
  }
}

}  // namespace cmix
