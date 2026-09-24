// The model: everything that is learned and kept in a state file.
//
//   predict()      - const: every specialist votes, the mixers combine the
//                    votes, two APM stages refine the result
//   learn()        - update every table, mixer and APM from the real bit
//   advance_byte() - move the context forward over a finished byte
//
// Keeping learn() and advance_byte() separate is what lets generation walk
// forward through its own output without teaching itself that output.
//
// Specialists (inputs to the mixers):
//   O0, D1, A (order 2), D3, D4, D5, D6, D8  order-N contexts
//   W1, W2                                   words (text)
//   C1, C2                                   byte classes (learned MDBE flags)
//   I1..I6                                   image: neighbouring pixels (image)
//   S1..S4                                   audio: earlier samples (audio)
//   E..                                      grid views, two votes each
//   B1, B2                                   long match
//   L                                        LSTM (level 2, optional)
//   G                                        graph of words / sound shapes / pixel runs (optional)
//   N                                        spiking network on that graph (optional)
//   bias
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/serial.h"
#include "model/adaptive.h"
#include "model/config.h"
#include "model/context_table.h"
#include "model/grid.h"
#include "model/history.h"
#include "model/lstm.h"
#include "graph/token_graph.h"
#include "model/match_model.h"
#include "model/mixer.h"
#include "model/stream.h"

namespace cmix {

// Which grid view is predicting best right now. Each view's cost is a
// decayed average of bits spent by its best vote; "none" (index 0) is the
// cost of the order-2 context, the no-grid baseline. The winner selects a
// mixer weight set, so each kind of region learns its own trust.
class ViewTracker {
 public:
  explicit ViewTracker(int views) : cost_(views + 1, 1.0f), input_cost_(2 * views, 1.0f) {}
  int best() const { return best_; }  // 0 = none, 1..views = view index + 1
  void learn(int baseline_p, const int* grid_p, int bit);
  void save(Writer& w) const;
  void load(Reader& r);

 private:
  std::vector<float> cost_, input_cost_;
  int best_ = 0;
};

class Model {
 public:
  static constexpr int kMaxInputs = 48;
  static constexpr int kMixers = 3;

  // Everything predict() works out that learn() needs, plus per-input
  // probabilities for reports (demo, compare).
  struct Votes {
    int n = 0;                     // number of inputs
    float x[kMaxInputs] = {};      // votes in stretch space
    int p[kMaxInputs] = {};        // each vote as a probability (16-bit)
    uint64_t ctx[kMaxInputs] = {}; // context hash for table-backed inputs
    int match_slot = -1;
    int sel[kMixers] = {};         // weight set per mixer
    float z[kMixers] = {};         // mixer outputs, stretch space
    int final_set = 0;
    float zf = 0;                  // final mixer output
    uint32_t apm_slot[2] = {};
    int mixed = 32768;             // final P(bit = 1), 16-bit
    float snn_raw = 0;             // text N vote before the readout (for learning it)
  };

  explicit Model(const Config& cfg);

  const Config& config() const { return cfg_; }
  int num_inputs() const { return n_inputs_; }
  std::string input_name(int i) const;   // short: "A", "D3", "W1", "E0u", "B1", ...
  std::string input_label(int i) const;  // longer, for reports
  const Grid& grid() const { return grid_; }
  int best_view() const { return tracker_.best(); }

  int predict(const Stream& s, BitPos bp, Votes& v) const;
  void learn(const Stream& s, BitPos bp, const Votes& v, int bit);
  // Byte-level learning (the LSTM); call when byte b is complete, before advance_byte.
  void learn_byte(const Stream& s, uint8_t b);
  void advance_byte(Stream& s, uint8_t b);
  // Graph learning; call after advance_byte of a learned byte.
  void learned_byte(const Stream& s);
  // Marks the start of a new record (an image or a sound) at the current
  // position, so pixel and sample positions count from here.
  void begin_record(Stream& s) const;

  // Graph planning (generation): at the start of a word / slice / run the
  // generator may pick which token to aim for.
  bool graph_can_plan(const Stream& s) const;
  std::vector<TokenGraph::Candidate> graph_candidates(const Stream& s) const;
  // Candidates for planning: graph_candidates, and for text without the
  // last two words (no "the the") and with very common words weakened
  // (weight / sqrt(word frequency)), so plans favour specific continuations.
  std::vector<TokenGraph::Candidate> plan_candidates(const Stream& s) const;
  void replan(Stream& s, Token t) const;  // kNoToken: nothing to plan
  // Steering toward the plan for the next byte: multiplies weight[c] for
  // bytes that move toward it. Text: the byte continuing the planned word.
  // Audio: louder or quieter high bytes, toward the planned slice's loudness.
  // Image: values near the planned run's brightness and colour.
  void plan_bias(const Stream& s, double strength, double* weight) const;
  const TokenGraph* graph() const { return graph_.get(); }
  const Vocab& vocab() const { return vocab_; }
  // Audio slice length in samples.
  int slice_samples() const { return cfg_.sample_rate > 0 ? std::max(16, cfg_.sample_rate / 50) : 160; }

  History& history() { return hist_; }
  const History& history() const { return hist_; }
  const Mixer& mixer(int k) const { return mix_[(size_t)k]; }
  const Mixer& final_mixer() const { return final_; }

  // Running totals over everything this model has learned from.
  uint64_t bytes_learned = 0;
  double bits_spent = 0;  // sum of -log2 P(real bit) while learning
  double recent_bpb = 0;  // bits/byte over roughly the last 4096 bytes learned
  void note_byte_cost(double bits) {
    const double rate = std::max(1.0 / 4096.0, 1.0 / (double)(bytes_learned + 1));
    recent_bpb += rate * (bits - recent_bpb);
  }

  void save(Writer& w) const;
  void load(Reader& r);

 private:
  void contexts(const Stream& s, uint64_t* out) const;  // byte-level hashes of table inputs
  void image_contexts(const Stream& s, uint64_t* out) const;
  void audio_contexts(const Stream& s, uint64_t* out) const;
  void graph_advance(Stream& s, uint8_t b) const;
  void text_graph_tree(GraphState& g) const;
  // SNN (model_snn.cpp): one token step (leak, then the given neurons spike).
  void snn_step(Stream& s, const Token* fire, const float* strength, int k) const;
  void snn_text_tree(const GraphState& g, SnnState& n) const;
  void snn_learn(const SnnState& n, Token actual);  // three-factor learning at a finished unit
  void snn_track_change(double unit_surprise);      // change detection (step 3b)
  static constexpr double kSnnRate = 0.1;
  static constexpr uint64_t kSnnMaxUnitBytes = 32;  // surprise is checked at least this often
  // SNN learning state (saved): surprise of the unit in progress, its
  // running baseline, and the text readout's gain and offset.
  double snn_unit_bits_ = 0;
  uint64_t snn_unit_bytes_ = 0;
  double snn_baseline_ = 0;
  float snn_gain_ = 1.0f, snn_offset_ = 0.0f;
  // Change detection: fast and slow averages of unit surprise; a jump opens
  // a "new region" for a few units. snn_changes_ counts detected jumps.
  double snn_fast_ = 0, snn_slow_ = 0;
  int snn_region_ = 0;
  uint64_t snn_changes_ = 0;

 public:
  // Units left in the current "new region" (0 = none) and jumps seen so far.
  int snn_region() const { return snn_region_; }
  uint64_t snn_changes() const { return snn_changes_; }

 private:
  bool snn_role_token(Token t) const;
  // Image: the run above acts through its own "above" neuron.
  static constexpr Token kAboveRole = 0x40000000u;
  int typed_first_ = 0;  // first image/audio input
  int graph_table_input_ = -1;  // audio/image G (table-backed)
  int graph_vote_input_ = -1;   // text G (direct vote)
  int snn_table_input_ = -1;    // audio/image N (table-backed)
  int snn_vote_input_ = -1;     // text N (direct vote)

  Config cfg_;
  History hist_;
  Grid grid_;
  // Input layout (indices into Votes arrays).
  int n_ctx_ = 0;       // table-backed inputs come first
  int grid_first_ = 0;  // first grid input
  int match_first_ = 0;
  int lstm_input_ = -1;  // -1 = no LSTM
  int bias_ = 0;
  int n_inputs_ = 0;
  ContextTable table_;
  MatchModel match_;
  ViewTracker tracker_;
  std::vector<Mixer> mix_;
  Mixer final_;
  Apm apm1_, apm2_;
  std::unique_ptr<Lstm> lstm_;
  std::unique_ptr<TokenGraph> graph_;
  Vocab vocab_;
};

}  // namespace cmix
