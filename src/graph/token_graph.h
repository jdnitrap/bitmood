// The graph specialist's memory: nodes are tokens (words, sound shapes,
// pixel-run shapes), edges are "this came next", with a count.
//
//   order-1 edges: token t0 -> next
//   order-2 edges: (t1, t0) -> next          (for images: (above, left) -> this)
//
// An edge is only a *fact* once it has been seen `confirm` times; before
// that it is a candidate and does not vote ("needs a yes before it is a
// fact"). Each node keeps at most kMaxEdges edges; when full, the weakest
// candidate makes room. Counts are halved when a node gets very busy, so
// old habits fade slowly.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/serial.h"

namespace cmix {

using Token = uint32_t;
constexpr Token kNoToken = 0xFFFFFFFFu;

class TokenGraph {
 public:
  static constexpr size_t kMaxEdges = 64;
  struct Edge {
    Token to;
    uint32_t count;
    float w = 1.0f;  // synapse weight (SNN): learned by three-factor learning
  };
  struct Candidate {
    Token token;
    double weight;
  };

  explicit TokenGraph(uint32_t confirm = 2) : confirm_(confirm) {}
  uint32_t confirm() const { return confirm_; }

  // Records that `next` followed t0 (and (t1, t0) when t1 is known).
  void add(Token t1, Token t0, Token next);
  // Confirmed next tokens after (t1, t0), order-2 edges weighted double.
  std::vector<Candidate> candidates(Token t1, Token t0) const;
  // The single most likely confirmed next token, or kNoToken.
  Token best(Token t1, Token t0) const;

  // For reports.
  size_t nodes() const { return out_.size(); }
  size_t edges(bool confirmed_only) const;
  // Order-1 edges of t0 (all, strongest first).
  std::vector<Edge> after(Token t0) const;
  // Order-1 edges of t0 in stored order, or nullptr (SNN spike spreading).
  const std::vector<Edge>* out(Token t0) const;
  // The order-1 edge from -> to, or nullptr (SNN learning).
  Edge* edge(Token from, Token to);
  // Records an order-1 edge from a role token (e.g. "the run above") only.
  void add_order1(Token from, Token next) { if (from != kNoToken && next != kNoToken) bump(key1(from), next); }
  // Order-1 edges pointing to `t` as (from, count), strongest first.
  std::vector<Edge> before(Token t) const;
  // Strongest confirmed order-1 edges overall: (from, to, count).
  struct Link {
    Token from, to;
    uint32_t count;
  };
  std::vector<Link> strongest(size_t n) const;

  void save(Writer& w) const;
  void load(Reader& r);

 private:
  static uint64_t key1(Token t0) { return (uint64_t(1) << 63) | t0; }
  static uint64_t key2(Token t1, Token t0) { return ((uint64_t)t1 << 32 | t0) & ~(uint64_t(1) << 63); }
  void bump(uint64_t key, Token next);
  uint32_t confirm_;
  std::unordered_map<uint64_t, std::vector<Edge>> out_;
};

// Names for text tokens (word hash -> word), and how often each word was seen.
class Vocab {
 public:
  void add(Token t, const std::string& word);
  const std::string* word(Token t) const;
  uint32_t count(Token t) const;
  size_t size() const { return words_.size(); }
  // Most frequent words.
  std::vector<std::pair<Token, uint32_t>> top(size_t n) const;
  void save(Writer& w) const;
  void load(Reader& r);

 private:
  struct Entry {
    std::string word;
    uint32_t count = 0;
  };
  std::unordered_map<Token, Entry> words_;
};

// Token for a (lowercase) word.
Token word_token(const char* w, int len);

}  // namespace cmix
