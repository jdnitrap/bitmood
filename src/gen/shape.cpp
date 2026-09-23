#include "gen/shape.h"

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace cmix {

bool is_word_byte(uint8_t b) { return std::isalpha(b) || b == '\'' || b >= 0x80; }

static uint8_t fold(uint8_t b) { return (b < 0x80) ? (uint8_t)std::tolower(b) : b; }

// ---- LineState ------------------------------------------------------------------

void LineState::accept(uint8_t b) {
  if (b == '\n') {
    col = 0;
    ++line;
    word.clear();
    last_word.clear();
    return;
  }
  ++col;
  if (is_word_byte(b)) {
    word.push_back((char)fold(b));
  } else if (!word.empty()) {
    last_word = word;
    word.clear();
  }
}

std::string LineState::ending() const {
  const std::string& w = word.empty() ? last_word : word;
  std::string letters;
  for (char c : w)
    if (std::isalpha((unsigned char)c)) letters.push_back(c);
  return letters.size() <= 2 ? letters : letters.substr(letters.size() - 2);
}

// ---- LineStartFilter ------------------------------------------------------------

void LineStartFilter::restrict(ByteMask& allowed) const {
  if (st_.col != 0) return;
  std::string ok = chars_;
  if (acrostic_) {
    const int i = st_.line - first_line_;
    if (i < 0 || i >= (int)chars_.size()) return;
    ok = std::string(1, chars_[(size_t)i]);
  }
  ByteMask keep{};
  for (unsigned char c : ok) {
    keep[c] = true;
    if (acrostic_) keep[std::toupper(c)] = keep[std::tolower(c)] = true;  // either case
  }
  for (int c = 0; c < 256; ++c) allowed[c] = allowed[c] && keep[c];
}

// ---- MaxLineFilter --------------------------------------------------------------

void MaxLineFilter::restrict(ByteMask& allowed) const {
  if (st_.col < max_) return;
  for (int c = 0; c < 256; ++c)
    if (c != '\n') allowed[c] = false;
}

// ---- WordListFilter -------------------------------------------------------------

void WordListFilter::Trie::add(const std::string& w) {
  int n = 0;
  for (unsigned char ch : w) {
    uint8_t c = fold(ch);
    auto it = nodes[(size_t)n].next.find(c);
    if (it == nodes[(size_t)n].next.end()) {
      nodes.push_back(Node());
      int id = (int)nodes.size() - 1;
      nodes[(size_t)n].next[c] = id;
      n = id;
    } else {
      n = it->second;
    }
  }
  nodes[(size_t)n].end = true;
}

std::shared_ptr<const WordListFilter::Trie> WordListFilter::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open word list " + path);
  auto t = std::make_shared<Trie>();
  std::string w;
  size_t count = 0;
  while (in >> w) {
    // Keep only the word bytes, so "hello," in a list still counts as "hello".
    std::string clean;
    for (unsigned char c : w)
      if (is_word_byte(c)) clean.push_back((char)c);
    if (!clean.empty()) {
      t->add(clean);
      ++count;
    }
  }
  if (count == 0) throw std::runtime_error("word list " + path + " has no words");
  return t;
}

void WordListFilter::restrict(ByteMask& allowed) const {
  if (free_) return;
  const auto& node = trie_->nodes[(size_t)(in_word_ ? node_ : 0)];
  for (int c = 0; c < 256; ++c) {
    if (!allowed[c]) continue;
    if (is_word_byte((uint8_t)c)) {
      allowed[c] = node.next.count(fold((uint8_t)c)) > 0;
    } else {
      static const std::string kBetween = " \n.,;:!?-\"()";
      if (kBetween.find((char)c) == std::string::npos) allowed[c] = false;
      else if (in_word_) allowed[c] = node.end;  // a word may only end when it is complete
    }
  }
}

void WordListFilter::accept(uint8_t b) {
  if (!is_word_byte(b)) {
    in_word_ = false;
    free_ = false;
    node_ = 0;
    return;
  }
  if (free_) return;
  const auto& node = trie_->nodes[(size_t)(in_word_ ? node_ : 0)];
  auto it = node.next.find(fold(b));
  in_word_ = true;
  if (it == node.next.end()) free_ = true;  // only possible for prompt text
  else node_ = it->second;
}

// ---- RhymeFilter ----------------------------------------------------------------

void RhymeFilter::restrict(ByteMask& allowed) const {
  if (target_.empty() || st_.col >= give_up_) return;
  const bool rhymes = st_.ending() == target_;
  if (!rhymes) {
    allowed['\n'] = false;
  } else if (st_.word.empty() && st_.col >= min_col_) {
    // The rhyming word is finished: end the line here.
    for (int c = 0; c < 256; ++c)
      if (c != '\n') allowed[c] = false;
  }
}

void RhymeFilter::accept(uint8_t b) {
  if (b == '\n') {
    // Line 0 of a pair sets the target for line 1; line 1 clears it.
    target_ = (st_.line % 2 == 0) ? st_.ending() : std::string();
  }
  st_.accept(b);
}

}  // namespace cmix
