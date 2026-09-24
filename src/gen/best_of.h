// Best-of-N: write several candidate lines from the same starting point,
// score each, keep the best, continue from there.
//
// Score = how close the candidate's surprise (bits/byte under the model) is
// to the surprise of real training text (lower is too predictable and
// repetitive), minus penalties for nonsense (spikes: single bytes the model
// found extremely unlikely) and for long verbatim copies of training text,
// plus a bonus for goals the constraints could not force (e.g. a rhyme),
// minus a penalty for lines much shorter or longer than the training
// text's typical line.
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
  double goal_weight = 1.0;   // weight of constraint goals met (e.g. a rhyme)
  double typical_len = 0;     // typical line length of the training text (0 = ignore length)
};

double score_candidate(const Candidate& c, const BestOfOptions& o);

// Generates one unit (line) with best-of-N and emits the winner into g.
// `unit` makes each unit's candidates use different random streams.
// Average line length of a text (bytes per newline), clamped to 10..200.
double typical_line_length(const std::vector<uint8_t>& text);

Candidate best_of_line(Generator& g, const BestOfOptions& o, std::shared_ptr<const SuffixArray> training,
                       uint64_t seed, uint64_t unit);

// For each slice, write several candidates and keep the one whose loudness and
// zero-crossing rate sit closest to the training audio (channel 0).
void best_of_audio(Generator& g, int candidates, int slice_bytes, int channels, double target_rms,
                   double target_zc, uint64_t seed, long long nbytes);

}  // namespace cmix
