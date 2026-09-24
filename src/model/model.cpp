#include "model/model.h"

#include <cctype>
#include <cmath>
#include <stdexcept>

#include "core/hash.h"
#include "core/math.h"
#include "model/byte_class.h"

namespace cmix {

namespace {

constexpr int kOrderInputs = Stream::kOrders;  // O0 D1 A D3 D4 D5 D6 D8
const char* const kOrderNames[kOrderInputs] = {"O0", "D1", "A", "D3", "D4", "D5", "D6", "D8"};
const char* const kOrderLabels[kOrderInputs] = {"O0 order-0", "D1 order-1", "A recent (order-2)", "D3 order-3",
                                                "D4 order-4", "D5 order-5", "D6 order-6",         "D8 order-8"};
constexpr int kInputA = 2;  // order-2: the no-grid baseline for the view tracker

// Update limit per table input: low orders see so much data they can
// average over long stretches; everything else adapts faster.
int limit_for(int input) { return input < 2 ? 1023 : 127; }

bool is_letter(uint8_t b) { return std::isalpha(b) || b >= 0x80; }

float mixer_lr(uint64_t bytes) { return std::max(0.005f, 0.05f / (1.0f + (float)bytes / 1024.0f)); }

}  // namespace

// ---- ViewTracker ------------------------------------------------------------

void ViewTracker::learn(int baseline_p, const int* grid_p, int bit) {
  static constexpr float kRate = 1.0f / 1024.0f;  // ~128 bytes of memory
  auto track = [&](float& c, int p) { c += kRate * ((float)bit_cost(p, bit) - c); };
  track(cost_[0], baseline_p);
  for (size_t i = 0; i < input_cost_.size(); ++i) track(input_cost_[i], grid_p[i]);
  best_ = 0;
  for (size_t v = 1; v < cost_.size(); ++v) {
    cost_[v] = std::min(input_cost_[2 * (v - 1)], input_cost_[2 * (v - 1) + 1]);
    if (cost_[v] < cost_[(size_t)best_]) best_ = (int)v;
  }
}

void ViewTracker::save(Writer& w) const {
  w.vec_f32(cost_);
  w.vec_f32(input_cost_);
  w.i32(best_);
}

void ViewTracker::load(Reader& r) {
  size_t a = cost_.size(), b = input_cost_.size();
  r.vec_f32(cost_);
  r.vec_f32(input_cost_);
  if (cost_.size() != a || input_cost_.size() != b) throw std::runtime_error("state file: view count mismatch");
  best_ = r.i32();
}

// ---- Model ------------------------------------------------------------------

Model::Model(const Config& cfg)
    : cfg_(cfg),
      hist_(size_t(1) << cfg.history_bits),
      grid_(cfg.views()),
      n_ctx_(kOrderInputs + (cfg.words() ? 2 : 0) + 2 + cfg.image_inputs() + cfg.audio_inputs() +
             cfg.graph_table_inputs() + cfg.snn_table_inputs() + grid_.num_inputs()),
      grid_first_(n_ctx_ - grid_.num_inputs()),
      match_first_(n_ctx_),
      lstm_input_(cfg.lstm_cells > 0 ? n_ctx_ + MatchModel::kInputs : -1),
      bias_(n_ctx_ + MatchModel::kInputs + (cfg.lstm_cells > 0 ? 1 : 0) + cfg.graph_vote_inputs() +
            cfg.snn_vote_inputs()),
      n_inputs_(bias_ + 1),
      table_(cfg.table_bits),
      low_(18),
      tracker_(grid_.num_views()),
      final_(kMixers + 1, 8, 0.3f),
      blend_(5, 8, 0.0f),
      apm1_(256),
      apm2_(257 * 256),
      apm3_(4 * 256) {
  if (n_inputs_ > kMaxInputs) throw std::runtime_error("too many model inputs");
  if (cfg.lstm_cells > 0) lstm_ = std::make_unique<Lstm>(cfg.lstm_cells);
  typed_first_ = grid_first_ - cfg.snn_table_inputs() - cfg.graph_table_inputs() - cfg.image_inputs() -
                 cfg.audio_inputs();
  if (cfg.graph_table_inputs()) graph_table_input_ = grid_first_ - 1 - cfg.snn_table_inputs();
  if (cfg.snn_table_inputs()) snn_table_input_ = grid_first_ - 1;
  if (cfg.graph_vote_inputs()) graph_vote_input_ = bias_ - 1 - cfg.snn_vote_inputs();
  if (cfg.snn_vote_inputs()) snn_vote_input_ = bias_ - 1;
  if (cfg.snn && !cfg.graph) throw std::runtime_error("--snn needs --graph (the network runs on the graph)");
  if (cfg.snn && !(cfg.snn_leak > 0.0f && cfg.snn_leak < 1.0f)) throw std::runtime_error("--snn-leak must be between 0 and 1");
  if (cfg.graph) {
    if (cfg.type == DataType::Raw) throw std::runtime_error("--graph works with text, image and audio memories");
    if (cfg.graph_confirm < 1 || cfg.graph_confirm > 1000) throw std::runtime_error("--graph-confirm must be 1..1000");
    graph_ = std::make_unique<TokenGraph>((uint32_t)cfg.graph_confirm);
  }
  if (cfg.type == DataType::Image && (cfg.width <= 0 || cfg.channels <= 0))
    throw std::runtime_error("image model needs width and channels");
  if (cfg.type == DataType::Audio && cfg.channels <= 0) throw std::runtime_error("audio model needs channels");
  mix_.emplace_back(n_inputs_, 4 * 8, 0.2f);                // match state x bit position
  mix_.emplace_back(n_inputs_, 257, 0.2f);                  // previous byte
  // winning grid view x bit position, and a copy used after a surprise jump
  mix_.emplace_back(n_inputs_, (grid_.num_views() + 1) * 8 * 2, 0.2f);
  // (mix + bit-so-far + 2 * previous-byte) / 4. Match refinement starts at 0.
  const float blend_row[5] = {0.25f, 0.25f, 0.50f, 0.0f, 0.0f};
  for (int set = 0; set < 8; ++set)
    for (int i = 0; i < 5; ++i) blend_.set(i, set, blend_row[i]);
}

std::string Model::input_name(int i) const {
  if (i < kOrderInputs) return kOrderNames[i];
  int k = kOrderInputs;
  if (cfg_.words()) {
    if (i == k) return "W1";
    if (i == k + 1) return "W2";
    k += 2;
  }
  if (i == k) return "C1";
  if (i == k + 1) return "C2";
  if (i == graph_table_input_ || i == graph_vote_input_) return "G";
  if (i == snn_table_input_ || i == snn_vote_input_) return "N";
  if (i >= typed_first_ && i < grid_first_)
    return (cfg_.type == DataType::Image ? "I" : "S") + std::to_string(i - typed_first_ + 1);
  if (i >= grid_first_ && i < match_first_) {
    int g = i - grid_first_;
    return "E" + std::to_string(g / 2) + (g % 2 ? "r" : "u");
  }
  if (i == match_first_) return "B1";
  if (i == match_first_ + 1) return "B2";
  if (i == lstm_input_) return "L";
  return "bias";
}

std::string Model::input_label(int i) const {
  if (i < kOrderInputs) return kOrderLabels[i];
  int k = kOrderInputs;
  if (cfg_.words()) {
    if (i == k) return "W word";
    if (i == k + 1) return "W word pair";
    k += 2;
  }
  if (i == k) return "C byte classes";
  if (i == k + 1) return "C classes + column";
  if (i == graph_table_input_ || i == graph_vote_input_)
    return cfg_.type == DataType::Text ? "G word graph" : (cfg_.type == DataType::Image ? "G pixel-run graph" : "G sound-shape graph");
  if (i == snn_table_input_ || i == snn_vote_input_) return "N spiking graph";
  if (i >= typed_first_ && i < grid_first_) {
    static const char* image[6] = {"I left pixel",       "I pixel above",       "I left+above average",
                                   "I gradient L+U-UL",  "I neighbourhood",     "I colour / vertical trend"};
    static const char* audio[4] = {"S last sample", "S linear trend", "S curve trend", "S level"};
    const int j = i - typed_first_;
    return cfg_.type == DataType::Image ? image[j] : audio[j];
  }
  if (i >= grid_first_ && i < match_first_) {
    int g = i - grid_first_;
    return "E " + view_name(grid_.view(g / 2)) + (g % 2 ? " (row)" : " (above)");
  }
  if (i == match_first_) return "B long match (learned)";
  if (i == match_first_ + 1) return "B long match (length)";
  if (i == lstm_input_) return "L LSTM (" + std::to_string(cfg_.lstm_cells) + " cells)";
  return "bias";
}

void Model::contexts(const Stream& s, uint64_t* out) const {
  int k = 0;
  for (int i = 0; i < kOrderInputs; ++i) out[k++] = s.order_hash[i];
  if (cfg_.words()) {
    out[k++] = hash_add(0x5701, s.word);
    out[k++] = hash_add(hash_add(0x5702, s.word), s.prev_word);
  }
  const uint64_t col = std::min<uint64_t>(hist_.end() - s.line_start, 63);
  out[k++] = hash_add(0xC101, s.classes & 0xFFF);
  out[k++] = hash_add(hash_add(0xC202, s.classes & 0xFF), col);
  if (cfg_.type == DataType::Image) image_contexts(s, out + k);
  if (cfg_.type == DataType::Audio) audio_contexts(s, out + k);
  k += cfg_.image_inputs() + cfg_.audio_inputs();
  if (graph_table_input_ >= 0) {
    // The shape the graph expects for this slice / run, with where we are in it.
    const uint64_t rel = hist_.end() - std::min(hist_.end(), s.record_start);
    uint64_t where;
    if (cfg_.type == DataType::Image) {
      const uint64_t c = (uint64_t)cfg_.channels;
      const int left = (rel >= c && hist_.has(hist_.end() - c)) ? hist_.at(hist_.end() - c) : 0;
      where = (rel % c) << 8 | (uint64_t)(left >> 5);
    } else {
      where = (rel & 1) << 8 | (uint64_t)(s.graph.n * 4 / (uint32_t)slice_samples());
    }
    out[k++] = hash_add(hash_add(0x6A00, (uint64_t)s.graph.expect), where);
    if (snn_table_input_ >= 0) out[k++] = hash_add(hash_add(0x5E00, (uint64_t)s.snn.predicted), where);
  }
  grid_.contexts(hist_, s, out + k);
}

namespace {
inline int clamp255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }
}  // namespace

void Model::image_contexts(const Stream& s, uint64_t* out) const {
  const uint64_t pos = hist_.end();
  const uint64_t rel = pos - std::min(pos, s.record_start);
  const int c = cfg_.channels, row = cfg_.row_bytes();
  // Byte `back` positions ago in this image, or 0 before its start.
  auto px = [&](uint64_t back) -> int {
    return (back >= 1 && back <= rel && hist_.has(pos - back)) ? hist_.at(pos - back) : 0;
  };
  const uint64_t ch = rel % (uint64_t)c;
  const uint64_t x = (rel / (uint64_t)c) % (uint64_t)cfg_.width;
  const int W = px((uint64_t)c), N = px((uint64_t)row), NW = px((uint64_t)(row + c));
  const int NE = x + 1 < (uint64_t)cfg_.width ? px((uint64_t)(row - c)) : N;
  const int NN = px(2 * (uint64_t)row);
  out[0] = hash_add(hash_add(0x1A1, (uint64_t)W), ch);
  out[1] = hash_add(hash_add(0x1A2, (uint64_t)N), ch);
  out[2] = hash_add(hash_add(0x1A3, (uint64_t)((W + N + 1) / 2)), ch);
  out[3] = hash_add(hash_add(0x1A4, (uint64_t)clamp255(W + N - NW)), ch);
  out[4] = hash_add(hash_add(hash_add(hash_add(0x1A5, (uint64_t)(W >> 3)), (uint64_t)(N >> 3)), (uint64_t)(NE >> 3)), ch);
  // Colour: this channel follows the previous channel's change from the left pixel.
  // Grey (or the first channel): the vertical trend 2N - NN.
  const int trend = ch > 0 ? clamp255(px(1) + W - px((uint64_t)c + 1)) : clamp255(2 * N - NN);
  out[5] = hash_add(hash_add(0x1A6, (uint64_t)trend), ch);
}

void Model::audio_contexts(const Stream& s, uint64_t* out) const {
  const uint64_t pos = hist_.end();
  const uint64_t rel = pos - std::min(pos, s.record_start);
  const uint64_t frame = 2 * (uint64_t)cfg_.channels;
  const uint64_t hi = rel & 1;  // 0: low byte next, 1: high byte next (little-endian)
  const uint64_t ch = (rel / 2) % (uint64_t)cfg_.channels;
  const uint64_t sample_start = rel - hi;  // relative position of this sample's low byte
  // Earlier samples of the same channel (k = 1, 2, 3 frames back), 0 before the start.
  auto sample = [&](uint64_t k) -> int {
    if (sample_start < k * frame) return 0;
    const uint64_t at = pos - (rel - (sample_start - k * frame));
    if (!hist_.has(at) || !hist_.has(at + 1)) return 0;
    return (int16_t)(hist_.at(at) | (hist_.at(at + 1) << 8));
  };
  const int s1 = sample(1), s2 = sample(2), s3 = sample(3);
  const int pred[3] = {s1, 2 * s1 - s2, 3 * s1 - 3 * s2 + s3};
  for (int k = 0; k < 3; ++k) {
    const int p = std::max(-32768, std::min(32767, pred[k]));
    uint64_t v;
    if (!hi) {
      v = (uint64_t)(p & 0xFF);  // expected low byte
    } else {
      const int low = hist_.back(0);  // the low byte just decided
      v = (uint64_t)(((p - low + 128) >> 8) & 0xFF);  // expected high byte given it
    }
    out[k] = hash_add(hash_add(hash_add(0x5A0 + (uint64_t)k, v), hi), ch);
  }
  out[3] = hash_add(hash_add(hash_add(0x5A4, hi), ch), (uint64_t)((s1 + 32768) >> 10));
}

int Model::predict(const Stream& s, BitPos bp, Votes& v) const {
  v.n = n_inputs_;
  uint64_t cx[kMaxInputs];
  contexts(s, cx);
  for (int i = 0; i < n_ctx_; ++i) {
    v.ctx[i] = hash_add(cx[i] + (uint64_t)i * 0x9E3779B97F4A7C15ull, bp.key());
    const ContextTable& table = i < 2 ? low_ : table_;
    const int64_t slot = table.lookup(v.ctx[i]);
    if (slot < 0 || table.n((uint32_t)slot) == 0) {
      v.p[i] = 32768;  // never seen: no opinion
      v.x[i] = 0.0f;
    } else {
      v.p[i] = table.p((uint32_t)slot);
      v.x[i] = stretch(v.p[i]);
    }
  }
  match_.predict(hist_, s, bp, v.x + match_first_, v.p + match_first_, v.match_slot);
  if (lstm_) {
    v.p[lstm_input_] = lstm_->p_bit(s.lstm, bp.index, bp.partial);
    v.x[lstm_input_] = s.lstm.ready ? stretch(v.p[lstm_input_]) : 0.0f;
  }
  // Tree-based votes (text G and N): P(bit) from a next-byte prefix-sum tree.
  auto tree_vote = [&](int input, const float* tree, bool ready) {
    if (input < 0) return;
    if (ready) {
      const int id = (int)bp.key();
      v.p[input] = to_p16(clampf(tree[2 * id + 1] / tree[id], 1e-4f, 1.0f - 1e-4f));
      v.x[input] = stretch(v.p[input]);
    } else {
      v.p[input] = 32768;
      v.x[input] = 0.0f;
    }
  };
  tree_vote(graph_vote_input_, s.graph.tree, s.graph.tree_ready);
  tree_vote(snn_vote_input_, s.snn.tree, s.snn.tree_ready);
  if (snn_vote_input_ >= 0 && s.snn.tree_ready) {
    // Readout: a learned gain and offset turn the charges' vote into N.
    v.snn_raw = v.x[snn_vote_input_];
    v.x[snn_vote_input_] = snn_gain_ * v.snn_raw + snn_offset_;
    v.p[snn_vote_input_] = to_p16(squash(v.x[snn_vote_input_]));
  } else {
    v.snn_raw = 0;
  }
  v.x[bias_] = 0.5f;
  v.p[bias_] = 32768;

  v.sel[0] = MatchModel::state(hist_, s, bp) * 8 + bp.index;
  v.sel[1] = s.last_byte + 1;
  // The odd set is the copy opened by a surprise jump (byte cost or the SNN).
  v.sel[2] = (tracker_.best() * 8 + bp.index) * 2 + ((adapt_region_ > 0 || snn_region_ > 0) ? 1 : 0);
  float xf[kMixers + 1];
  for (int k = 0; k < kMixers; ++k) {
    v.z[k] = clampf(mix_[(size_t)k].dot(v.x, v.sel[k]), -15.0f, 15.0f);
    xf[k] = v.z[k];
  }
  xf[kMixers] = 0.5f;
  v.final_set = bp.index;
  v.zf = clampf(final_.dot(xf, v.final_set), -15.0f, 15.0f);
  const int pmix = to_p16(squash(v.zf));

  // Three refinements of the mixed probability: bits so far, previous byte,
  // and match state. A small mixer learns how to blend them (per bit).
  const int a1 = apm1_.refine(pmix, bp.key(), v.apm_slot[0]);
  const int a2 = apm2_.refine(pmix, (size_t)(s.last_byte + 1) * 256 + bp.key(), v.apm_slot[1]);
  const int a3 = apm3_.refine(pmix, (size_t)(v.sel[0] / 8) * 256 + bp.key(), v.apm_slot[2]);
  // Probability space, so the initial weights 1/4, 1/4, 1/2, 0 reproduce
  // (pmix + a1 + 2*a2) / 4. The mixer then moves those weights per bit.
  v.blend_x[0] = pmix / 65536.0f;
  v.blend_x[1] = a1 / 65536.0f;
  v.blend_x[2] = a2 / 65536.0f;
  v.blend_x[3] = a3 / 65536.0f;
  v.blend_x[4] = 0.0f;
  v.blend_set = bp.index;
  v.blend_z = clampf(blend_.dot(v.blend_x, v.blend_set), 1.0f / 65536.0f, 1.0f - 1.0f / 65536.0f);
  v.mixed = to_p16(v.blend_z);
  return v.mixed;
}

void Model::learn(const Stream& s, BitPos bp, const Votes& v, int bit) {
  (void)s;
  const float target = (float)bit;
  float lr = mixer_lr(bytes_learned);
  // The copied weight set has no history of its own yet, so it learns faster.
  if (adapt_region_ > 0 || snn_region_ > 0) lr = std::min(0.05f, std::max(lr * 4.0f, 0.02f));
  float xf[kMixers + 1];
  for (int k = 0; k < kMixers; ++k) {
    mix_[(size_t)k].learn(v.x, v.sel[k], target - squash(v.z[k]), lr);
    xf[k] = v.z[k];
  }
  xf[kMixers] = 0.5f;
  final_.learn(xf, v.final_set, target - squash(v.zf), 0.002f);
  // Slow on purpose: a faster rate walked off the (1, 1, 2) blend and cost
  // about 2 KB on a 750 KB novel.
  blend_.learn(v.blend_x, v.blend_set, target - v.blend_z, 0.00002f);

  for (int i = 0; i < n_ctx_; ++i) {
    ContextTable& table = i < 2 ? low_ : table_;
    // Orders 6 and 8 take a slot only on the second sighting, so a hash seen
    // once cannot evict a useful one. Orders 0 and 1 live in low_. Shorter
    // orders, words, classes and the grid repeat often enough to claim now.
    const bool second_sight = i >= 6 && i < kOrderInputs;
    int64_t slot = table.lookup(v.ctx[i]);
    if (slot < 0) {
      if (second_sight && !table.promote(v.ctx[i])) continue;
      slot = table.claim(v.ctx[i]);
    }
    table.update((uint32_t)slot, bit, limit_for(i));
  }
  if (cfg_.snn) {
    snn_unit_bits_ += bit_cost(v.mixed, bit);
    if (bp.index == 7 && ++snn_unit_bytes_ >= kSnnMaxUnitBytes) {
      // No token finished for a while (numbers, symbols, binary): still
      // watch surprise so change detection isn't blind there.
      snn_track_change(snn_unit_bits_ / snn_unit_bytes_);
      snn_unit_bits_ = 0;
      snn_unit_bytes_ = 0;
    }
    if (snn_vote_input_ >= 0 && v.snn_raw != 0) {
      // Readout learning: the same error rule as the mixers.
      const float err = target - squash(v.x[snn_vote_input_]);
      snn_gain_ = clampf(snn_gain_ + 0.002f * err * v.snn_raw, 0.0f, 4.0f);
      snn_offset_ = clampf(snn_offset_ + 0.002f * err, -2.0f, 2.0f);
    }
  }
  match_.learn(v.match_slot, bit);
  apm1_.update(v.apm_slot[0], bit);
  apm2_.update(v.apm_slot[1], bit);
  apm3_.update(v.apm_slot[2], bit);
  tracker_.learn(v.p[kInputA], v.p + grid_first_, bit);
}

void Model::learn_byte(const Stream& s, uint8_t b) {
  if (lstm_) lstm_->learn(s.lstm, b);
}

void Model::begin_record(Stream& s) {
  audio_prev_ = 0;
  s.record_start = hist_.end();
  // A new image / sound starts a new slice or run.
  s.graph.n = 0;
  s.graph.sum_sq = 0;
  s.graph.zero_crossings = 0;
  s.graph.last_sign = 0;
  s.graph.plan = GraphState::kNone;
  s.graph.planned = false;
  s.graph.expect = GraphState::kNone;
}

void Model::advance_byte(Stream& s, uint8_t b) {
  if (lstm_) lstm_->forward(s.lstm, b);
  if (hist_.push(b)) match_.rebuild(hist_);
  s.last_byte = b;
  ++s.bytes;
  if (b == '\n') {
    s.prev_line_start = s.line_start;
    s.line_start = hist_.end();
  }
  s.widths.update(hist_);
  for (int i = 0; i < Stream::kOrders; ++i) {
    uint64_t x = 0x0D00 + (uint64_t)i;
    for (int j = 0; j < Stream::kOrderLen[i] && (size_t)j < hist_.size(); ++j) x = hash_add(x, hist_.back((size_t)j));
    s.order_hash[i] = x;
  }
  if (is_letter(b)) {
    s.word = hash_add(s.word ? s.word : 0x3000, (uint64_t)std::tolower(b));
  } else if (s.word) {
    s.prev_word = s.word;
    s.word = 0;
  }
  s.classes = ((s.classes << 4) | byte_class(b)) & 0xFFFF;
  match_.advance(hist_, s);
  if (graph_) graph_advance(s, b);
}

void Model::save(Writer& w) const {
  w.tag("HIST");
  w.u64(hist_.base());
  w.vec_u8(hist_.data());
  w.tag("TABL");
  table_.save(w);
  w.tag("LOWT");
  low_.save(w);
  w.tag("MTCH");
  match_.save(w);
  w.tag("TRAK");
  tracker_.save(w);
  w.tag("MIXR");
  for (const Mixer& m : mix_) m.save(w);
  final_.save(w);
  w.tag("BLND");
  blend_.save(w);
  w.tag("APMS");
  apm1_.save(w);
  apm2_.save(w);
  apm3_.save(w);
  if (lstm_) {
    w.tag("LSTM");
    lstm_->save(w);
  }
  if (cfg_.snn) {
    w.tag("SNNL");
    w.f64(snn_unit_bits_);
    w.u64(snn_unit_bytes_);
    w.f64(snn_baseline_);
    w.f32(snn_gain_);
    w.f32(snn_offset_);
    w.f64(snn_fast_);
    w.f64(snn_slow_);
    w.i32(snn_region_);
    w.u64(snn_changes_);
  }
  if (graph_) {
    w.tag("GRPH");
    graph_->save(w);
    vocab_.save(w);
  }
  w.tag("STAT");
  w.u64(bytes_learned);
  w.f64(bits_spent);
  w.f64(recent_bpb);
  w.f64(adapt_fast_);
  w.f64(adapt_slow_);
  w.i32(adapt_region_);
  w.i32(adapt_cool_);
  w.f64(audio_sum_sq_);
  w.u64(audio_n_);
  w.u64(audio_zc_);
  w.i32(audio_prev_);
}

void Model::load(Reader& r) {
  r.expect_tag("HIST");
  uint64_t base = r.u64();
  std::vector<uint8_t> bytes;
  r.vec_u8(bytes);
  if (bytes.size() > hist_.cap()) throw std::runtime_error("state file: history larger than configured");
  hist_.assign(std::move(bytes), base);
  match_.rebuild(hist_);
  r.expect_tag("TABL");
  table_.load(r);
  r.expect_tag("LOWT");
  low_.load(r);
  r.expect_tag("MTCH");
  match_.load(r);
  r.expect_tag("TRAK");
  tracker_.load(r);
  r.expect_tag("MIXR");
  for (Mixer& m : mix_) m.load(r);
  final_.load(r);
  r.expect_tag("BLND");
  blend_.load(r);
  r.expect_tag("APMS");
  apm1_.load(r);
  apm2_.load(r);
  apm3_.load(r);
  if (lstm_) {
    r.expect_tag("LSTM");
    lstm_->load(r);
  }
  if (cfg_.snn) {
    r.expect_tag("SNNL");
    snn_unit_bits_ = r.f64();
    snn_unit_bytes_ = r.u64();
    snn_baseline_ = r.f64();
    snn_gain_ = r.f32();
    snn_offset_ = r.f32();
    snn_fast_ = r.f64();
    snn_slow_ = r.f64();
    snn_region_ = r.i32();
    snn_changes_ = r.u64();
  }
  if (graph_) {
    r.expect_tag("GRPH");
    graph_->load(r);
    vocab_.load(r);
  }
  r.expect_tag("STAT");
  bytes_learned = r.u64();
  bits_spent = r.f64();
  recent_bpb = r.f64();
  adapt_fast_ = r.f64();
  adapt_slow_ = r.f64();
  adapt_region_ = r.i32();
  adapt_cool_ = r.i32();
  audio_sum_sq_ = r.f64();
  audio_n_ = r.u64();
  audio_zc_ = r.u64();
  audio_prev_ = r.i32();
}

void Model::note_byte_cost(double bits) {
  const double rate = std::max(1.0 / 4096.0, 1.0 / (double)(bytes_learned + 1));
  recent_bpb += rate * (bits - recent_bpb);
  if (bytes_learned == 0) {
    adapt_fast_ = adapt_slow_ = bits;
  } else {
    adapt_fast_ += (bits - adapt_fast_) / 32.0;
    adapt_slow_ += (bits - adapt_slow_) / 2048.0;
  }
  if (adapt_region_ > 0) {
    --adapt_region_;
    return;
  }
  if (adapt_cool_ > 0) {
    --adapt_cool_;
    return;
  }
  // A short average well above the long one means the file changed character.
  // Twice the long-run cost, plus 1.5 bits: a new kind of data, not a hard sentence.
  if (snn_region_ > 0 || bytes_learned < 4096) return;
  if (adapt_fast_ > adapt_slow_ * 2.0 + 1.5) {
    adapt_region_ = 2048;
    adapt_cool_ = 4096;
    open_region(true);
  }
}

void Model::open_region(bool forget) {
  Mixer& m = mix_[2];
  const int groups = grid_.num_views() + 1;
  for (int g = 0; g < groups; ++g)
    for (int b = 0; b < 8; ++b) m.copy_set((g * 8 + b) * 2, (g * 8 + b) * 2 + 1);
  if (!forget) return;
  table_.halve_counts();
  low_.halve_counts();
}

double Model::audio_rms() const {
  return audio_n_ > 0 ? std::sqrt(audio_sum_sq_ / (double)audio_n_) : 0.0;
}

double Model::audio_zc_rate() const {
  return audio_n_ > 0 ? (double)audio_zc_ / (double)audio_n_ : 0.0;
}

void Model::note_audio_sample(const Stream& s) {
  if (cfg_.type != DataType::Audio || cfg_.channels <= 0) return;
  const uint64_t end = hist_.end();
  const uint64_t rel = end - std::min(end, s.record_start);
  if (rel < 2) return;
  const uint64_t frame = 2ull * (uint64_t)cfg_.channels;
  if ((rel - 1) % frame != 1) return;  // channel 0, high byte just written
  if (!hist_.has(end - 2) || !hist_.has(end - 1)) return;
  const int16_t sample = (int16_t)(hist_.at(end - 2) | (hist_.at(end - 1) << 8));
  audio_sum_sq_ += (double)sample * (double)sample;
  const int sign = sample > 0 ? 1 : (sample < 0 ? -1 : 0);
  if (sign != 0 && audio_prev_ != 0 && sign != audio_prev_) ++audio_zc_;
  if (sign != 0) audio_prev_ = sign;
  ++audio_n_;
}

int Model::preferred_byte(const Stream& s) const {
  if (cfg_.type == DataType::Image && cfg_.width > 0 && cfg_.channels > 0) {
    const uint64_t pos = hist_.end();
    const uint64_t rel = pos - std::min(pos, s.record_start);
    const int c = cfg_.channels;
    const int row = cfg_.row_bytes();
    auto px = [&](uint64_t back) -> int {
      return (back >= 1 && back <= rel && hist_.has(pos - back)) ? hist_.at(pos - back) : 0;
    };
    const int W = px((uint64_t)c), N = px((uint64_t)row), NW = px((uint64_t)(row + c));
    return clamp255(W + N - NW);
  }
  if (cfg_.type != DataType::Audio || cfg_.channels <= 0 || hist_.empty()) return -1;
  const uint64_t pos = hist_.end();
  const uint64_t rel = pos - std::min(pos, s.record_start);
  if ((rel & 1) == 0) return -1;  // low byte: leave it to the model
  const uint64_t frame = 2ull * (uint64_t)cfg_.channels;
  if ((rel / 2) % (uint64_t)cfg_.channels != 0) return -1;
  const uint64_t sample_start = rel - 1;
  auto sample = [&](uint64_t k) -> int {
    if (sample_start < k * frame) return 0;
    const uint64_t at = pos - (rel - (sample_start - k * frame));
    if (!hist_.has(at) || !hist_.has(at + 1)) return 0;
    return (int16_t)(hist_.at(at) | (hist_.at(at + 1) << 8));
  };
  const int pred = std::max(-32768, std::min(32767, 2 * sample(1) - sample(2)));
  return ((pred - hist_.back(0) + 128) >> 8) & 0xFF;
}

void Model::sample_bias(const Stream& s, double* weight) const {
  const int prefer = preferred_byte(s);
  if (prefer < 0) return;
  for (int c = 0; c < 256; ++c) {
    const double d = (c - prefer) / 32.0;
    weight[c] *= std::exp(-0.5 * d * d);
  }
}

}  // namespace cmix
