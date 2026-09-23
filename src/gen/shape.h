// Constraints on the *shape* of generated text: how lines start and end,
// which words are allowed, and rhyme. See constraints.h for the interface.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gen/constraints.h"

namespace cmix {

// Tracks the current line: column and the letters of the last word.
struct LineState {
  int col = 0;           // bytes since the last newline
  int line = 0;          // lines finished so far
  int prev_len = 0;      // length of the previous line
  std::string word;      // letters of the word in progress (lowercase ASCII)
  std::string last_word; // last finished word on this line
  void accept(uint8_t b);
  // Ending used for rhyme: last 2 letters of the word in progress, else of the last word.
  std::string ending() const;
};

bool is_word_byte(uint8_t b);  // ASCII letter, apostrophe, or UTF-8 byte

// Each line must start with one of `chars`, or, with an acrostic word,
// line i must start with its i-th letter (then free once the word is used up).
class LineStartFilter : public Constraint {
 public:
  LineStartFilter(std::string chars, bool acrostic) : chars_(std::move(chars)), acrostic_(acrostic) {}
  std::unique_ptr<Constraint> clone() const override { return std::make_unique<LineStartFilter>(*this); }
  void restrict(ByteMask& allowed) const override;
  void accept(uint8_t b) override { st_.accept(b); }
  // The acrostic counts lines from the first line the output starts.
  void begin_output() override { first_line_ = st_.line + (st_.col > 0 ? 1 : 0); }

 private:
  std::string chars_;
  bool acrostic_;
  LineState st_;
  int first_line_ = 0;
};

// Lines end by column `max`: at `max` only a newline may follow. Soft, so a
// multi-byte character in progress is finished first.
class MaxLineFilter : public Constraint {
 public:
  explicit MaxLineFilter(int max) : max_(max) {}
  std::unique_ptr<Constraint> clone() const override { return std::make_unique<MaxLineFilter>(*this); }
  void restrict(ByteMask& allowed) const override;
  void accept(uint8_t b) override { st_.accept(b); }
  bool soft() const override { return true; }
  int priority() const override { return 30; }

 private:
  int max_;
  LineState st_;
};

// Only words from a list (case-insensitive for ASCII). A word is a run of
// word bytes; it may only end (be followed by a non-word byte) once it is
// a complete word from the list. Between words only spaces, newlines and
// ordinary punctuation are allowed, so nothing but listed words appears.
class WordListFilter : public Constraint {
 public:
  struct Trie {
    struct Node {
      std::map<uint8_t, int> next;
      bool end = false;
    };
    std::vector<Node> nodes{Node()};
    void add(const std::string& w);
  };
  explicit WordListFilter(std::shared_ptr<const Trie> t) : trie_(std::move(t)) {}
  static std::shared_ptr<const Trie> load(const std::string& path);
  std::unique_ptr<Constraint> clone() const override { return std::make_unique<WordListFilter>(*this); }
  void restrict(ByteMask& allowed) const override;
  void accept(uint8_t b) override;
  bool must_continue() const override { return in_word_ && !free_ && !trie_->nodes[(size_t)node_].end; }

 private:
  std::shared_ptr<const Trie> trie_;
  int node_ = 0;       // position in the trie for the word in progress
  bool in_word_ = false;
  bool free_ = false;  // word in progress is not in the list (came from the prompt)
};

// Rough rhyme, AABB. On the second line of each pair, as soon as a word
// ending like the first line's last word (last two letters) is finished
// past column `min_col`, the line ends there. It never forbids ending a
// line (that only pushes the model into text it doesn't believe); instead
// satisfaction() tells best-of-N which candidate lines rhymed, so
// `--rhyme --best-of N` picks rhyming lines out of natural ones.
class RhymeFilter : public Constraint {
 public:
  explicit RhymeFilter(int min_col = 20) : min_col_(min_col) {}
  std::unique_ptr<Constraint> clone() const override { return std::make_unique<RhymeFilter>(*this); }
  void restrict(ByteMask& allowed) const override;
  void accept(uint8_t b) override;
  bool soft() const override { return true; }
  int priority() const override { return 20; }
  double satisfaction() const override { return score_; }

 private:
  int min_col_;
  double score_ = 0;  // +1 per second line that rhymed, -1 per one that didn't
  LineState st_;
  std::string target_;  // ending the current line must match (empty = free line)
};

}  // namespace cmix
