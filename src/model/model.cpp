#include "model/model.h"

#include <cctype>
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
             grid_.num_inputs()),
      grid_first_(n_ctx_ - grid_.num_inputs()),
      match_first_(n_ctx_),
      bias_(n_ctx_ + MatchModel::kInputs),
      n_inputs_(bias_ + 1),
      table_(cfg.table_bits),
      tracker_(grid_.num_views()),
      final_(kMixers + 1, 8, 0.3f),
      apm1_(256),
      apm2_(257 * 256) {
  if (n_inputs_ > kMaxInputs) throw std::runtime_error("too many model inputs");
  typed_first_ = grid_first_ - cfg.image_inputs() - cfg.audio_inputs();
  if (cfg.type == DataType::Image && (cfg.width <= 0 || cfg.channels <= 0))
    throw std::runtime_error("image model needs width and channels");
  if (cfg.type == DataType::Audio && cfg.channels <= 0) throw std::runtime_error("audio model needs channels");
  mix_.emplace_back(n_inputs_, 4 * 8, 0.2f);                         // match state x bit position
  mix_.emplace_back(n_inputs_, 257, 0.2f);                           // previous byte
  mix_.emplace_back(n_inputs_, (grid_.num_views() + 1) * 8, 0.2f);  // winning grid view x bit position
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
  if (i >= typed_first_ && i < grid_first_)
    return (cfg_.type == DataType::Image ? "I" : "S") + std::to_string(i - typed_first_ + 1);
  if (i >= grid_first_ && i < match_first_) {
    int g = i - grid_first_;
    return "E" + std::to_string(g / 2) + (g % 2 ? "r" : "u");
  }
  if (i == match_first_) return "B1";
  if (i == match_first_ + 1) return "B2";
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
    const int64_t slot = table_.lookup(v.ctx[i]);
    if (slot < 0 || table_.n((uint32_t)slot) == 0) {
      v.p[i] = 32768;  // never seen: no opinion
      v.x[i] = 0.0f;
    } else {
      v.p[i] = table_.p((uint32_t)slot);
      v.x[i] = stretch(v.p[i]);
    }
  }
  match_.predict(hist_, s, bp, v.x + match_first_, v.p + match_first_, v.match_slot);
  v.x[bias_] = 0.5f;
  v.p[bias_] = 32768;

  v.sel[0] = MatchModel::state(hist_, s, bp) * 8 + bp.index;
  v.sel[1] = s.last_byte + 1;
  v.sel[2] = tracker_.best() * 8 + bp.index;
  float xf[kMixers + 1];
  for (int k = 0; k < kMixers; ++k) {
    v.z[k] = clampf(mix_[(size_t)k].dot(v.x, v.sel[k]), -15.0f, 15.0f);
    xf[k] = v.z[k];
  }
  xf[kMixers] = 0.5f;
  v.final_set = bp.index;
  v.zf = clampf(final_.dot(xf, v.final_set), -15.0f, 15.0f);
  const int pmix = to_p16(squash(v.zf));

  // Refine with two APMs: order 0 (bits so far) and order 1 (previous byte too).
  const int a1 = apm1_.refine(pmix, bp.key(), v.apm_slot[0]);
  const int a2 = apm2_.refine(pmix, (size_t)(s.last_byte + 1) * 256 + bp.key(), v.apm_slot[1]);
  v.mixed = clampi((pmix + a1 + 2 * a2 + 2) / 4, kProbMin, kProbMax);
  return v.mixed;
}

void Model::learn(const Stream& s, BitPos bp, const Votes& v, int bit) {
  (void)s;
  (void)bp;
  const float target = (float)bit;
  const float lr = mixer_lr(bytes_learned);
  float xf[kMixers + 1];
  for (int k = 0; k < kMixers; ++k) {
    mix_[(size_t)k].learn(v.x, v.sel[k], target - squash(v.z[k]), lr);
    xf[k] = v.z[k];
  }
  xf[kMixers] = 0.5f;
  final_.learn(xf, v.final_set, target - squash(v.zf), 0.002f);

  for (int i = 0; i < n_ctx_; ++i) table_.update(table_.claim(v.ctx[i]), bit, limit_for(i));
  match_.learn(v.match_slot, bit);
  apm1_.update(v.apm_slot[0], bit);
  apm2_.update(v.apm_slot[1], bit);
  tracker_.learn(v.p[kInputA], v.p + grid_first_, bit);
}

void Model::advance_byte(Stream& s, uint8_t b) {
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
}

void Model::save(Writer& w) const {
  w.tag("HIST");
  w.u64(hist_.base());
  w.vec_u8(hist_.data());
  w.tag("TABL");
  table_.save(w);
  w.tag("MTCH");
  match_.save(w);
  w.tag("TRAK");
  tracker_.save(w);
  w.tag("MIXR");
  for (const Mixer& m : mix_) m.save(w);
  final_.save(w);
  w.tag("APMS");
  apm1_.save(w);
  apm2_.save(w);
  w.tag("STAT");
  w.u64(bytes_learned);
  w.f64(bits_spent);
  w.f64(recent_bpb);
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
  r.expect_tag("MTCH");
  match_.load(r);
  r.expect_tag("TRAK");
  tracker_.load(r);
  r.expect_tag("MIXR");
  for (Mixer& m : mix_) m.load(r);
  final_.load(r);
  r.expect_tag("APMS");
  apm1_.load(r);
  apm2_.load(r);
  r.expect_tag("STAT");
  bytes_learned = r.u64();
  bits_spent = r.f64();
  recent_bpb = r.f64();
}

}  // namespace cmix
