#include "graph/token_graph.h"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "core/hash.h"

namespace cmix {

namespace {
constexpr uint32_t kBusy = 1u << 20;  // node total that triggers halving
}

Token word_token(const char* w, int len) {
  uint64_t h = 0x77;
  for (int i = 0; i < len; ++i) h = hash_add(h, (uint8_t)w[i]);
  Token t = (Token)(h >> 32);
  return t == kNoToken ? 0 : t;
}

void TokenGraph::bump(uint64_t key, Token next) {
  std::vector<Edge>& e = out_[key];
  for (Edge& x : e) {
    if (x.to != next) continue;
    ++x.count;
    uint64_t total = 0;
    for (const Edge& y : e) total += y.count;
    if (total > kBusy)
      for (Edge& y : e) y.count = (y.count + 1) / 2;
    return;
  }
  if (e.size() < kMaxEdges) {
    e.push_back({next, 1, 1.0f});
    return;
  }
  // Full: the weakest edge (a candidate that never got its yes) makes room.
  auto weakest = std::min_element(e.begin(), e.end(), [](const Edge& a, const Edge& b) {
    return a.count < b.count || (a.count == b.count && a.to < b.to);
  });
  if (weakest->count < confirm_) *weakest = {next, 1, 1.0f};
}

void TokenGraph::add(Token t1, Token t0, Token next) {
  if (t0 == kNoToken || next == kNoToken) return;
  bump(key1(t0), next);
  if (t1 != kNoToken) bump(key2(t1, t0), next);
}

std::vector<TokenGraph::Candidate> TokenGraph::candidates(Token t1, Token t0) const {
  std::map<Token, double> w;  // ordered: results don't depend on hash order
  auto collect = [&](uint64_t key, double scale) {
    auto it = out_.find(key);
    if (it == out_.end()) return;
    for (const Edge& e : it->second)
      if (e.count >= confirm_) w[e.to] += scale * e.count;
  };
  if (t0 != kNoToken) collect(key1(t0), 1.0);
  if (t1 != kNoToken && t0 != kNoToken) collect(key2(t1, t0), 2.0);
  std::vector<Candidate> out;
  for (const auto& kv : w) out.push_back({kv.first, kv.second});
  return out;
}

Token TokenGraph::best(Token t1, Token t0) const {
  Token b = kNoToken;
  double bw = 0;
  for (const Candidate& c : candidates(t1, t0))
    if (c.weight > bw) b = c.token, bw = c.weight;
  return b;
}

size_t TokenGraph::edges(bool confirmed_only) const {
  size_t n = 0;
  for (const auto& kv : out_)
    for (const Edge& e : kv.second) n += !confirmed_only || e.count >= confirm_;
  return n;
}

std::vector<TokenGraph::Edge> TokenGraph::after(Token t0) const {
  auto it = out_.find(key1(t0));
  if (it == out_.end()) return {};
  std::vector<Edge> e = it->second;
  std::sort(e.begin(), e.end(), [](const Edge& a, const Edge& b) { return a.count > b.count || (a.count == b.count && a.to < b.to); });
  return e;
}

const std::vector<TokenGraph::Edge>* TokenGraph::out(Token t0) const {
  auto it = out_.find(key1(t0));
  return it == out_.end() ? nullptr : &it->second;
}

TokenGraph::Edge* TokenGraph::edge(Token from, Token to) {
  auto it = out_.find(key1(from));
  if (it == out_.end()) return nullptr;
  for (Edge& e : it->second)
    if (e.to == to) return &e;
  return nullptr;
}

std::vector<TokenGraph::Edge> TokenGraph::before(Token t) const {
  std::vector<Edge> e;
  for (const auto& kv : out_) {
    if (!(kv.first >> 63)) continue;  // order-1 keys only
    for (const Edge& x : kv.second)
      if (x.to == t) e.push_back({(Token)(kv.first & 0xFFFFFFFFu), x.count, x.w});
  }
  std::sort(e.begin(), e.end(), [](const Edge& a, const Edge& b) { return a.count > b.count || (a.count == b.count && a.to < b.to); });
  return e;
}

std::vector<TokenGraph::Link> TokenGraph::strongest(size_t n) const {
  std::vector<Link> all;
  for (const auto& kv : out_) {
    if (!(kv.first >> 63)) continue;
    for (const Edge& e : kv.second)
      if (e.count >= confirm_) all.push_back({(Token)(kv.first & 0xFFFFFFFFu), e.to, e.count});
  }
  std::sort(all.begin(), all.end(), [](const Link& a, const Link& b) {
    return a.count > b.count || (a.count == b.count && (a.from < b.from || (a.from == b.from && a.to < b.to)));
  });
  if (all.size() > n) all.resize(n);
  return all;
}

void TokenGraph::save(Writer& w) const {
  w.u32(confirm_);
  std::vector<uint64_t> keys;
  for (const auto& kv : out_) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());  // deterministic file
  w.u64(keys.size());
  for (uint64_t k : keys) {
    const auto& e = out_.at(k);
    w.u64(k);
    w.u32((uint32_t)e.size());
    for (const Edge& x : e) {
      w.u32(x.to);
      w.u32(x.count);
      w.f32(x.w);
    }
  }
}

void TokenGraph::load(Reader& r) {
  confirm_ = r.u32();
  const uint64_t n = r.u64();
  if (n > r.remaining() / 12) throw std::runtime_error("state file: graph truncated");
  out_.clear();
  out_.reserve((size_t)n);
  for (uint64_t i = 0; i < n; ++i) {
    const uint64_t k = r.u64();
    const uint32_t m = r.u32();
    if (m > kMaxEdges) throw std::runtime_error("state file: bad graph node");
    std::vector<Edge>& e = out_[k];
    for (uint32_t j = 0; j < m; ++j) {
      Token to = r.u32();
      uint32_t c = r.u32();
      float sw = r.f32();
      e.push_back({to, c, sw});
    }
  }
}

// ---- Vocab ------------------------------------------------------------------

void Vocab::add(Token t, const std::string& word) {
  Entry& e = words_[t];
  if (e.word.empty()) e.word = word;
  ++e.count;
}

const std::string* Vocab::word(Token t) const {
  auto it = words_.find(t);
  return it == words_.end() ? nullptr : &it->second.word;
}

uint32_t Vocab::count(Token t) const {
  auto it = words_.find(t);
  return it == words_.end() ? 0 : it->second.count;
}

std::vector<std::pair<Token, uint32_t>> Vocab::top(size_t n) const {
  std::vector<std::pair<Token, uint32_t>> v;
  for (const auto& kv : words_) v.push_back({kv.first, kv.second.count});
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second || (a.second == b.second && a.first < b.first); });
  if (v.size() > n) v.resize(n);
  return v;
}

void Vocab::save(Writer& w) const {
  std::vector<Token> keys;
  for (const auto& kv : words_) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());
  w.u64(keys.size());
  for (Token t : keys) {
    const Entry& e = words_.at(t);
    w.u32(t);
    w.u32(e.count);
    w.u32((uint32_t)e.word.size());
    w.bytes(e.word.data(), e.word.size());
  }
}

void Vocab::load(Reader& r) {
  const uint64_t n = r.u64();
  if (n > r.remaining() / 12) throw std::runtime_error("state file: vocabulary truncated");
  words_.clear();
  for (uint64_t i = 0; i < n; ++i) {
    Token t = r.u32();
    Entry e;
    e.count = r.u32();
    const uint32_t len = r.u32();
    if (len > 256 || len > r.remaining()) throw std::runtime_error("state file: bad vocabulary entry");
    e.word.resize(len);
    r.bytes(&e.word[0], len);
    words_[t] = std::move(e);
  }
}

}  // namespace cmix
