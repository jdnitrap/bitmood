// Every byte the model has seen, oldest first, capped in size.
// Specialists read it to look back (long match, order-N hashes).
// Generation can roll it back with truncate() to try alternatives.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cmix {

class History {
 public:
  explicit History(size_t cap = size_t(1) << 20) : cap_(cap) {}

  size_t size() const { return buf_.size(); }
  bool empty() const { return buf_.empty(); }
  uint8_t operator[](size_t i) const { return buf_[i]; }
  // back(k): the byte k positions before the end (back(0) = newest). Caller checks k < size().
  uint8_t back(size_t k) const { return buf_[buf_.size() - 1 - k]; }

  // Appends a byte. When full, drops the oldest half; returns true if that happened
  // so owners of absolute positions (hash indexes) can rebuild.
  bool push(uint8_t b) {
    buf_.push_back(b);
    if (buf_.size() <= cap_) return false;
    buf_.erase(buf_.begin(), buf_.begin() + (std::ptrdiff_t)(buf_.size() / 2));
    ++drops_;
    return true;
  }

  // Roll back to an earlier length. Only valid if no drop happened since.
  void truncate(size_t n) { if (n < buf_.size()) buf_.resize(n); }
  uint64_t drops() const { return drops_; }
  size_t cap() const { return cap_; }

  const std::vector<uint8_t>& data() const { return buf_; }
  std::vector<uint8_t>& mutable_data() { return buf_; }

 private:
  std::vector<uint8_t> buf_;
  size_t cap_;
  uint64_t drops_ = 0;
};

}  // namespace cmix
