// The graph specialist G inside the model: how each data type is cut into
// tokens as the context moves (graph_advance), how the graph is learned
// (learned_byte), and planning for generation.
//
//   text   token = a finished word (lowercase letters)
//   audio  token = shape of a 20 ms slice of channel 0
//   image  token = shape of an 8-pixel run; its "previous" tokens are the
//          run above (t1) and the run to the left (t0)
#include <cctype>
#include <cmath>
#include <cstring>

#include "core/math.h"
#include "graph/shapes.h"
#include "model/model.h"

namespace cmix {

namespace {

constexpr Token kRowStart = kImageCodes;  // "left" token at the start of an image row

bool is_word_letter(uint8_t b) { return std::isalpha(b) || b >= 0x80; }

void clear_plan(GraphState& g) {
  g.plan = GraphState::kNone;
  g.planned = false;
}

void finish(GraphState& g, Token t1, Token t0, Token next) {
  g.completed = true;
  g.learn_t1 = t1;
  g.learn_t0 = t0;
  g.learn_next = next;
}

}  // namespace

void Model::text_graph_tree(GraphState& g) const {
  g.tree_ready = false;
  if (g.overflow) return;
  float w[256] = {};
  double total = 0;
  auto add_word = [&](const std::string& word, double weight) {
    if ((int)word.size() < g.wlen || std::memcmp(word.data(), g.word, (size_t)g.wlen) != 0) return false;
    int ch = (int)word.size() == g.wlen ? ' ' : (uint8_t)word[(size_t)g.wlen];
    if (g.wlen == 0 && g.capital && ch < 0x80) ch = std::toupper(ch);
    w[ch] += (float)weight;
    total += weight;
    return true;
  };
  for (const TokenGraph::Candidate& c : graph_->candidates(g.t1, g.t0))
    if (const std::string* word = vocab_.word(c.token)) add_word(*word, c.weight);
  // A planned word (generation) dominates the vote while it still fits.
  if (g.plan != GraphState::kNone)
    if (const std::string* word = vocab_.word(g.plan)) add_word(*word, 3.0 * std::max(1.0, total));
  if (total <= 0) return;
  for (int k = 0; k < 256; ++k) g.tree[256 + k] = 0.98f * (float)(w[k] / total) + 0.02f / 256.0f;
  for (int n = 255; n >= 1; --n) g.tree[n] = g.tree[2 * n] + g.tree[2 * n + 1];
  g.tree_ready = true;
}

void Model::graph_advance(Stream& s, uint8_t b) const {
  GraphState& g = s.graph;
  g.completed = false;
  const uint64_t rel = hist_.end() - std::min(hist_.end(), s.record_start);

  if (cfg_.type == DataType::Text) {
    if (is_word_letter(b)) {
      if (!g.overflow) {
        if (g.wlen < GraphState::kMaxWord) g.word[g.wlen++] = (char)(b < 0x80 ? std::tolower(b) : b);
        else g.overflow = true;
      }
      g.capital = false;
    } else {
      if (g.wlen > 0 && !g.overflow) {
        const Token t = word_token(g.word, g.wlen);
        finish(g, g.t1, g.t0, t);
        std::memcpy(g.done_word, g.word, (size_t)g.wlen);
        g.done_len = g.wlen;
        g.t1 = g.t0;
        g.t0 = t;
      }
      if (g.wlen > 0 || g.overflow) clear_plan(g);
      g.wlen = 0;
      g.overflow = false;
      if (b == '.' || b == '!' || b == '?' || b == '\n') g.capital = true;
    }
    text_graph_tree(g);
    return;
  }

  if (cfg_.type == DataType::Audio) {
    // A channel-0 sample just finished?
    if (rel == 0 || rel % 2 != 0 || (rel / 2 - 1) % (uint64_t)cfg_.channels != 0) return;
    const int v = (int16_t)(hist_.back(1) | (hist_.back(0) << 8));
    g.sum_sq += (double)v * v;
    const int sign = v > 0 ? 1 : (v < 0 ? -1 : 0);
    if (sign != 0) {
      if (g.last_sign != 0 && sign != g.last_sign) ++g.zero_crossings;
      g.last_sign = sign;
    }
    if (++g.n < (uint32_t)slice_samples()) return;
    const double rms = std::sqrt(g.sum_sq / g.n);
    const Token code = audio_code(rms, g.zero_crossings, g.prev_rms);
    finish(g, g.t1, g.t0, code);
    g.t1 = g.t0;
    g.t0 = code;
    g.prev_rms = rms;
    g.n = 0;
    g.sum_sq = 0;
    g.zero_crossings = 0;
    clear_plan(g);
    g.expect = graph_->best(g.t1, g.t0);
    return;
  }

  // Image: act at pixel boundaries only.
  const uint64_t c = (uint64_t)cfg_.channels, w = (uint64_t)cfg_.width;
  if (rel == 0 || rel % c != 0) return;
  const uint64_t pix = rel / c, x = pix % w, y = pix / w;
  if (x % kRunPixels != 0) return;
  auto code = [&](uint64_t ry, uint64_t x0, uint64_t x1) {
    return image_run_code(hist_, s.record_start, cfg_.width, cfg_.channels, ry, (int)x0, (int)x1);
  };
  // The run that just finished (the last run of the previous row when x == 0).
  const uint64_t ry = x == 0 ? y - 1 : y;
  const uint64_t xe = x == 0 ? w : x, xs = (xe - 1) / kRunPixels * kRunPixels;
  const Token done = code(ry, xs, xe);
  if (done != kNoToken) {
    const Token above = ry > 0 ? code(ry - 1, xs, xe) : kNoToken;
    const Token left = xs > 0 ? code(ry, xs - kRunPixels, xs) : kRowStart;
    finish(g, above, left, done);
    g.t1 = above;
    g.t0 = done;
  }
  clear_plan(g);
  // What the graph expects for the run starting now.
  const Token above = y > 0 ? code(y - 1, x, std::min(x + kRunPixels, w)) : kNoToken;
  const Token left = x > 0 ? code(y, x - kRunPixels, x) : kRowStart;
  g.expect = graph_->best(above, left);
}

void Model::learned_byte(const Stream& s) {
  if (!graph_ || !s.graph.completed) return;
  const GraphState& g = s.graph;
  graph_->add(g.learn_t1, g.learn_t0, g.learn_next);
  if (cfg_.type == DataType::Text)
    vocab_.add(g.learn_next, std::string(g.done_word, (size_t)g.done_len));
}

bool Model::graph_can_plan(const Stream& s) const {
  if (!graph_ || s.graph.planned) return false;
  const GraphState& g = s.graph;
  const uint64_t rel = hist_.end() - std::min(hist_.end(), s.record_start);
  switch (cfg_.type) {
    case DataType::Text: return g.wlen == 0 && !g.overflow;
    case DataType::Raw: return false;
    case DataType::Audio: return g.n == 0 && rel % 2 == 0;
    case DataType::Image: {
      const uint64_t c = (uint64_t)cfg_.channels;
      return rel % c == 0 && (rel / c) % (uint64_t)cfg_.width % kRunPixels == 0;
    }
  }
  return false;
}

std::vector<TokenGraph::Candidate> Model::graph_candidates(const Stream& s) const {
  if (!graph_) return {};
  const GraphState& g = s.graph;
  if (cfg_.type != DataType::Image) return graph_->candidates(g.t1, g.t0);
  // Image: the run above and the run to the left of the run starting now.
  const uint64_t rel = hist_.end() - std::min(hist_.end(), s.record_start);
  const uint64_t c = (uint64_t)cfg_.channels, w = (uint64_t)cfg_.width;
  const uint64_t pix = rel / c, x = pix % w, y = pix / w;
  auto code = [&](uint64_t ry, uint64_t x0, uint64_t x1) {
    return image_run_code(hist_, s.record_start, cfg_.width, cfg_.channels, ry, (int)x0, (int)x1);
  };
  const Token above = y > 0 ? code(y - 1, x, std::min(x + kRunPixels, w)) : kNoToken;
  const Token left = x > 0 ? code(y, x - kRunPixels, x) : kRowStart;
  return graph_->candidates(above, left);
}

int Model::plan_next_byte(const Stream& s) const {
  const GraphState& g = s.graph;
  if (cfg_.type != DataType::Text || g.plan == GraphState::kNone || g.overflow) return -1;
  const std::string* word = vocab_.word(g.plan);
  if (!word || (int)word->size() < g.wlen || std::memcmp(word->data(), g.word, (size_t)g.wlen) != 0) return -1;
  if ((int)word->size() == g.wlen) return ' ';
  int ch = (uint8_t)(*word)[(size_t)g.wlen];
  if (g.wlen == 0 && g.capital && ch < 0x80) ch = std::toupper(ch);
  return ch;
}

void Model::replan(Stream& s, Token t) const {
  GraphState& g = s.graph;
  g.planned = true;
  if (t == kNoToken) return;
  g.plan = t;
  if (cfg_.type == DataType::Text) text_graph_tree(g);
  else g.expect = t;
}

}  // namespace cmix
