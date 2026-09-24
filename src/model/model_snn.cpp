// The spiking network (SNN) on the graph: graph nodes are neurons, graph
// edges are synapses.
//
//   step:   every token step, all charge leaks (x snn_leak) and eligibility
//           traces fade; the neuron of the token that just finished SPIKES
//   spread: a spike adds charge to the neurons its edges point to,
//           charge += synapse weight x (edge count / all confirmed counts)
//   vote N: the charged neurons are the prediction (text: next-letter
//           probabilities from the charged words; audio/image: the
//           most-charged shape as a context)
//
// Because charge leaks slowly, tokens from several steps back still
// contribute, which the plain graph (last two tokens only) can't do.
// Only charged neurons are stored and touched: the work is event-driven.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

#include "graph/shapes.h"
#include "model/model.h"

namespace cmix {

namespace {
constexpr float kTraceDecay = 0.5f;
constexpr float kMinCharge = 0.01f;
}  // namespace

void Model::snn_step(Stream& s, const Token* fire, const float* strength, int k) const {
  SnnState& n = s.snn;
  // Snapshot for three-factor learning (what led up to the token just finished).
  n.last_predicted = n.predicted;
  n.last_nf = n.nf;
  std::copy(n.fired, n.fired + n.nf, n.last_fired);
  std::copy(n.trace, n.trace + n.nf, n.last_trace);
  // Leak, and forget neurons whose charge has faded.
  int kept = 0;
  for (int i = 0; i < n.n; ++i) {
    const float c = n.charge[i] * cfg_.snn_leak;
    if (c < kMinCharge) continue;
    n.tok[kept] = n.tok[i];
    n.charge[kept++] = c;
  }
  n.n = kept;
  // Eligibility traces fade.
  kept = 0;
  for (int i = 0; i < n.nf; ++i) {
    const float t = n.trace[i] * kTraceDecay;
    if (t < 0.05f) continue;
    n.fired[kept] = n.fired[i];
    n.trace[kept++] = t;
  }
  n.nf = kept;

  auto add_charge = [&](Token t, float c) {
    for (int i = 0; i < n.n; ++i)
      if (n.tok[i] == t) {
        n.charge[i] += c;
        return;
      }
    if (n.n < SnnState::kMaxCharged) {
      n.tok[n.n] = t;
      n.charge[n.n++] = c;
      return;
    }
    // Full: replace the weakest neuron if the new charge is larger.
    int weakest = 0;
    for (int i = 1; i < n.n; ++i)
      if (n.charge[i] < n.charge[weakest]) weakest = i;
    if (c > n.charge[weakest]) {
      n.tok[weakest] = t;
      n.charge[weakest] = c;
    }
  };

  for (int f = 0; f < k; ++f) {
    const Token t = fire[f];
    if (t == kNoToken) continue;
    // Remember the spike for learning (pre-synaptic trace).
    if (n.nf == SnnState::kMaxFired) {
      std::move(n.fired + 1, n.fired + n.nf, n.fired);
      std::move(n.trace + 1, n.trace + n.nf, n.trace);
      --n.nf;
    }
    n.fired[n.nf] = t;
    n.trace[n.nf++] = 1.0f;
    // Spread along confirmed edges.
    const std::vector<TokenGraph::Edge>* out = graph_->out(t);
    if (!out) continue;
    double total = 0;
    for (const TokenGraph::Edge& e : *out)
      if (e.count >= graph_->confirm()) total += e.count;
    if (total <= 0) continue;
    for (const TokenGraph::Edge& e : *out)
      if (e.count >= graph_->confirm()) add_charge(e.to, strength[f] * e.w * (float)(e.count / total));
  }

  // The prediction: the most-charged real token (role tokens are inputs only).
  n.predicted = SnnState::kNone;
  float best = 0;
  for (int i = 0; i < n.n; ++i)
    if (n.charge[i] > best && !snn_role_token(n.tok[i])) best = n.charge[i], n.predicted = n.tok[i];
}

void Model::snn_text_tree(const GraphState& g, SnnState& n) const {
  n.tree_ready = false;
  if (g.overflow) return;
  float w[256] = {};
  double total = 0;
  for (int i = 0; i < n.n; ++i) {
    const std::string* word = vocab_.word(n.tok[i]);
    if (!word || (int)word->size() < g.wlen || std::memcmp(word->data(), g.word, (size_t)g.wlen) != 0) continue;
    int ch = (int)word->size() == g.wlen ? ' ' : (uint8_t)(*word)[(size_t)g.wlen];
    if (g.wlen == 0 && g.capital && ch < 0x80) ch = std::toupper(ch);
    w[ch] += n.charge[i];
    total += n.charge[i];
  }
  if (total <= 0) return;
  for (int k = 0; k < 256; ++k) n.tree[256 + k] = 0.98f * (float)(w[k] / total) + 0.02f / 256.0f;
  for (int m = 255; m >= 1; --m) n.tree[m] = n.tree[2 * m] + n.tree[2 * m + 1];
  n.tree_ready = true;
}

void Model::snn_learn(const SnnState& n, Token actual) {
  // Surprise of the unit that just finished, against a running baseline.
  if (snn_unit_bytes_ == 0) return;
  const double u = snn_unit_bits_ / snn_unit_bytes_;
  snn_unit_bits_ = 0;
  snn_unit_bytes_ = 0;
  if (snn_baseline_ <= 0) snn_baseline_ = u;
  // "Dopamine": better than usual > 0, worse < 0.
  const double d = std::max(-1.0, std::min(1.0, (snn_baseline_ - u) / std::max(snn_baseline_, 0.5)));
  snn_baseline_ += (u - snn_baseline_) / 64.0;
  snn_track_change(u);
  if (actual == kNoToken) return;
  auto nudge = [&](Token from, Token to, double delta) {
    if (to == kNoToken) return;
    if (TokenGraph::Edge* e = graph_->edge(from, to))
      e->w = (float)std::max(0.05, std::min(4.0, e->w + delta));
  };
  for (int i = 0; i < n.last_nf; ++i) {
    const Token pre = n.last_fired[i];
    const double eligible = n.last_trace[i];
    // The connections that made the prediction: reinforced when it went
    // well, weakened when it went badly.
    nudge(pre, n.last_predicted, kSnnRate * eligible * d);
    // When surprised, the path to what really came is strengthened.
    if (n.last_predicted != actual && d < 0) nudge(pre, actual, kSnnRate * eligible * -d);
  }
}

void Model::snn_track_change(double u) {
  if (snn_slow_ <= 0) snn_fast_ = snn_slow_ = u;
  // Tuned on four joined files: finds file boundaries (within ~50-200
  // bytes) and real content changes (CSS -> JavaScript) without firing on
  // ordinary hard words.
  snn_fast_ += (u - snn_fast_) / 8.0;
  snn_slow_ += (u - snn_slow_) / 1024.0;
  if (snn_region_ > 0) --snn_region_;
  // A sudden rise in surprise: something new started. The region lasts
  // 128 units, and no new jump is reported during it.
  if (snn_region_ == 0 && snn_fast_ > snn_slow_ * 2.0 + 1.0) {
    snn_region_ = 128;
    ++snn_changes_;
    // Same weight-set copy the byte-level detector uses. Skip it when that
    // detector already opened the region for this stretch.
    if (adapt_region_ == 0) open_region(true);
    if (adapt_cool_ < 2048) adapt_cool_ = 2048;
  }
}

bool Model::snn_role_token(Token t) const {
  return cfg_.type == DataType::Image && (t & kAboveRole) != 0 && t != kNoToken;
}

}  // namespace cmix
