// Every byte the model has seen, oldest first, capped in size.
// Specialists read it to look back (long match, order-N hashes, grid views).
// Generation can roll it back with truncate() to try alternatives.
//
// Positions come in two kinds:
//   index    - 0..size()-1 into what is currently kept
//   absolute - counts every byte ever pushed; absolute = base() + index
// Absolute positions stay valid when old bytes are dropped.
#pragma once

#include <algorithm>
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

  uint64_t base() const { return base_; }
  uint64_t end() const { return base_ + buf_.size(); }  // absolute position of the next byte
  bool has(uint64_t abs) const { return abs >= base_ && abs < end(); }
  uint8_t at(uint64_t abs) const { return buf_[(size_t)(abs - base_)]; }

  // Appends a byte. When full, drops the oldest half; returns true if that happened
  // so owners of positions (hash indexes) can rebuild.
  bool push(uint8_t b) {
    buf_.push_back(b);
    if (buf_.size() <= cap_) return false;
    size_t drop = buf_.size() / 2;
    buf_.erase(buf_.begin(), buf_.begin() + (std::ptrdiff_t)drop);
    base_ += drop;
    return true;
  }

  // Drops old bytes if needed so that `room` more bytes fit without a drop.
  void make_room(size_t room) {
    if (buf_.size() + room <= cap_) return;
    size_t drop = std::min(buf_.size(), buf_.size() + room - cap_);
    buf_.erase(buf_.begin(), buf_.begin() + (std::ptrdiff_t)drop);
    base_ += drop;
  }

  // Roll back to an earlier absolute end. Only valid if no drop happened since.
  void truncate_to(uint64_t abs_end) {
    if (abs_end >= base_ && abs_end < end()) buf_.resize((size_t)(abs_end - base_));
  }
  size_t cap() const { return cap_; }

  const std::vector<uint8_t>& data() const { return buf_; }
  // For loading: replace contents and base together.
  void assign(std::vector<uint8_t> bytes, uint64_t base) {
    buf_ = std::move(bytes);
    base_ = base;
  }

 private:
  std::vector<uint8_t> buf_;
  size_t cap_;
  uint64_t base_ = 0;
};

}  // namespace cmix
