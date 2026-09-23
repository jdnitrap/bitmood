// Writing together with the model: the logic behind `cmix-bit write`,
// kept apart from the terminal so it can be tested on its own.
//
// Typed bytes move the model's context at once (so suggestions follow the
// cursor) but are only *learned* when the word is finished: at a space,
// punctuation, newline, or an accepted suggestion. Until then Backspace
// just rolls the context back; nothing learned ever has to be undone.
#pragma once

#include <cstdint>
#include <string>

#include "gen/generator.h"

namespace cmix {

class WritingSession {
 public:
  // `g` should be greedy (top_k = 1) for stable suggestions.
  WritingSession(Generator& g, size_t suggest_len);

  void type(uint8_t b);
  bool backspace();                     // false when there is nothing unlearned to delete
  void accept_suggestion(bool one_word);
  void commit();                        // learn the word in progress now

  const std::string& text() const { return text_; }            // everything written
  const std::string& suggestion() const { return suggestion_; } // next few bytes the model expects
  uint64_t learned() const { return learned_; }

 private:
  static bool ends_word(uint8_t b);
  void refresh();
  Generator& g_;
  size_t suggest_len_;
  Generator::Snapshot at_commit_;
  std::string text_, pending_, suggestion_;
  uint64_t learned_ = 0;
};

}  // namespace cmix
