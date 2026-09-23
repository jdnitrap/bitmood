// Best-of-N: write several candidate lines from the same starting point,
// score each, keep the best, continue from there.
//
// Score = how close the candidate's surprise (bits/byte under the model) is
// to the surprise of real training text (lower is too predictable and
// repetitive), minus penalties for nonsense (spikes: single bytes the model
// found extremely unlikely) and for long verbatim copies of training text.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "gen/generator.h"
#include "gen/suffix_array.h"

namespace cmix {

struct Candidate {
  std::vector<uint8_t> bytes;
  std::vector<double> bits;  // model cost of each byte
  double bpb = 0;
  size_t copy = 0;            // longest run copied from training text
  int spikes = 0;             // bytes costing more than BestOfOptions::spike_bits
  double score = 0;
};

struct BestOfOptions {
  int candidates = 8;
  size_t max_len = 200;       // a candidate ends at a newline or this many bytes
  double target_bpb = 2.5;    // surprise to aim for
  size_t copy_allowance = 12; // copies up to this long are not penalized
  double spike_bits = 9.0;    // a byte this surprising counts as a spike (1 in 512)
};

double score_candidate(const Candidate& c, const BestOfOptions& o);

// Generates one unit (line) with best-of-N and emits the winner into g.
// `unit` makes each unit's candidates use different random streams.
Candidate best_of_line(Generator& g, const BestOfOptions& o, std::shared_ptr<const SuffixArray> training,
                       uint64_t seed, uint64_t unit);

}  // namespace cmix
