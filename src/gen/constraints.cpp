#include "gen/constraints.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace cmix {

// ---- CharsetFilter ------------------------------------------------------------

CharsetFilter::Mode CharsetFilter::parse(const std::string& s) {
  if (s == "any") return Mode::Any;
  if (s == "ascii") return Mode::Ascii;
  if (s == "utf8") return Mode::Utf8;
  if (s == "seen") return Mode::Seen;
  throw std::runtime_error("--charset must be seen, utf8, ascii or any");
}

void CharsetFilter::restrict(ByteMask& allowed) const {
  if (mode_ == Mode::Any) return;
  for (int c = 0; c < 256; ++c) {
    bool ok;
    if (pending_ > 0) {
      ok = c >= lo_ && c <= hi_;
    } else {
      ok = (c >= 0x20 && c < 0x7F) || c == '\n' || c == '\t';
      // UTF-8 lead bytes (C0, C1, F5+ can never appear in valid UTF-8).
      if (mode_ != Mode::Ascii) ok = ok || (c >= 0xC2 && c <= 0xF4);
    }
    if (mode_ == Mode::Seen && !seen_[c]) ok = false;
    if (!ok) allowed[c] = false;
  }
}

void CharsetFilter::accept(uint8_t b) {
  if (pending_ > 0) {
    --pending_;
    lo_ = 0x80;
    hi_ = 0xBF;
    return;
  }
  // Ranges for the first continuation byte rule out overlong forms and surrogates.
  if (b >= 0xC2 && b <= 0xDF) pending_ = 1;
  else if (b == 0xE0) pending_ = 2, lo_ = 0xA0;
  else if (b == 0xED) pending_ = 2, hi_ = 0x9F;
  else if (b >= 0xE1 && b <= 0xEF) pending_ = 2;
  else if (b == 0xF0) pending_ = 3, lo_ = 0x90;
  else if (b == 0xF4) pending_ = 3, hi_ = 0x8F;
  else if (b >= 0xF1 && b <= 0xF3) pending_ = 3;
}

// ---- NoveltyFilter ------------------------------------------------------------

uint64_t NoveltyFilter::hash_run(const std::deque<uint8_t>& w, int extra) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (uint8_t b : w) h = (h ^ b) * 0x100000001b3ull;
  if (extra >= 0) h = (h ^ (uint64_t)extra) * 0x100000001b3ull;
  return h;
}

void NoveltyFilter::restrict(ByteMask& allowed) const {
  if (limit_ <= 0 || (int)window_.size() < limit_) return;
  // Own output: forbid bytes that would repeat an earlier (limit+1)-byte run.
  for (int c = 0; c < 256; ++c)
    if (allowed[c] && std::binary_search(seen_runs_.begin(), seen_runs_.end(), hash_run(window_, c)))
      allowed[c] = false;
  // Training text.
  if (!sa_ || sa_->empty()) return;
  std::vector<uint8_t> w(window_.begin(), window_.end());
  SuffixArray::Range r = sa_->find(w.data(), w.size());
  if (r.empty()) return;
  for (int c = 0; c < 256; ++c)
    if (allowed[c] && !sa_->extend(r, w.size(), (uint8_t)c).empty()) allowed[c] = false;
}

void NoveltyFilter::accept(uint8_t b) {
  if (limit_ <= 0) return;
  if ((int)window_.size() == limit_) {
    uint64_t h = hash_run(window_, b);
    auto it = std::lower_bound(seen_runs_.begin(), seen_runs_.end(), h);
    if (it == seen_runs_.end() || *it != h) seen_runs_.insert(it, h);
  }
  window_.push_back(b);
  if ((int)window_.size() > limit_) window_.pop_front();
}

// ---- CopyMeter ------------------------------------------------------------------

void CopyMeter::accept(uint8_t b) {
  tail_.push_back(b);
  if (tail_.size() > kCap) tail_.pop_front();
  // The copied run can grow by at most one; shrink until it is found again.
  size_t len = std::min(run_ + 1, tail_.size());
  std::vector<uint8_t> t(tail_.end() - (std::ptrdiff_t)len, tail_.end());
  while (len > 0 && !sa_->contains(t.data() + (t.size() - len), len)) --len;
  run_ = len;
  if (run_ > longest_) longest_ = run_;
}

}  // namespace cmix
