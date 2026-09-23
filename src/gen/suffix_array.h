// Suffix array over a byte string: every suffix, sorted. Lets us ask
// "does this exact byte sequence occur in the training text?" in
// O(len * log n), which is what the novelty cap needs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cmix {

class SuffixArray {
 public:
  SuffixArray() = default;
  explicit SuffixArray(std::vector<uint8_t> text);  // O(n log n) prefix doubling

  bool empty() const { return text_.empty(); }
  size_t size() const { return text_.size(); }

  // Range [lo, hi) of suffixes that start with pattern[0..n). Empty if absent.
  struct Range {
    size_t lo = 0, hi = 0;
    bool empty() const { return lo >= hi; }
  };
  Range find(const uint8_t* pattern, size_t n) const;
  // Narrows a range for pattern p (length n) to suffixes continuing with byte c.
  Range extend(Range r, size_t n, uint8_t c) const;
  bool contains(const uint8_t* pattern, size_t n) const { return !find(pattern, n).empty(); }

 private:
  // Byte at offset k of suffix i, or -1 past the end of the text.
  int at(size_t i, size_t k) const {
    size_t p = sa_[i] + k;
    return p < text_.size() ? text_[p] : -1;
  }
  std::vector<uint8_t> text_;
  std::vector<uint32_t> sa_;
};

}  // namespace cmix
