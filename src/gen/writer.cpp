#include "gen/writer.h"

#include <cctype>

namespace cmix {

WritingSession::WritingSession(Generator& g, size_t suggest_len)
    : g_(g), suggest_len_(suggest_len), at_commit_(g.snapshot()) {
  refresh();
}

bool WritingSession::ends_word(uint8_t b) { return b < 0x80 && !std::isalnum(b) && b != '\'' && b != '_'; }

void WritingSession::type(uint8_t b) {
  text_.push_back((char)b);
  pending_.push_back((char)b);
  g_.feed(b);
  if (ends_word(b) && !g_.must_continue()) commit();
  else refresh();
}

bool WritingSession::backspace() {
  if (pending_.empty()) return false;
  // Remove one whole UTF-8 character: continuation bytes, then its lead byte.
  do {
    pending_.pop_back();
    text_.pop_back();
  } while (!pending_.empty() && ((uint8_t)pending_.back() & 0xC0) == 0x80);
  if (!pending_.empty() && ((uint8_t)pending_.back() & 0xC0) == 0xC0) {
    pending_.pop_back();
    text_.pop_back();
  }
  g_.restore(at_commit_);
  for (unsigned char c : pending_) g_.feed(c);
  refresh();
  return true;
}

void WritingSession::accept_suggestion(bool one_word) {
  std::string take = suggestion_;
  if (one_word) {
    // Up to and including the first word break after some word bytes.
    size_t i = 0;
    while (i < take.size() && ends_word((uint8_t)take[i])) ++i;
    while (i < take.size() && !ends_word((uint8_t)take[i])) ++i;
    if (i < take.size()) ++i;
    take = take.substr(0, i);
  }
  for (unsigned char c : take) type(c);
  commit();
}

void WritingSession::commit() {
  if (!pending_.empty()) {
    g_.restore(at_commit_);
    for (unsigned char c : pending_) g_.learn_byte(c);
    learned_ += pending_.size();
    pending_.clear();
    g_.ensure_room();
    at_commit_ = g_.snapshot();
  }
  refresh();
}

void WritingSession::refresh() {
  const Generator::Snapshot here = g_.snapshot();
  suggestion_.clear();
  while (suggestion_.size() < suggest_len_ || (g_.must_continue() && suggestion_.size() < suggest_len_ + 4)) {
    uint8_t b = g_.next();
    if (b == '\n') break;
    suggestion_.push_back((char)b);
  }
  g_.restore(here);
}

}  // namespace cmix
