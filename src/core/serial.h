// Binary serialization with a running checksum.
// Values are written little-endian regardless of host, so state files
// move between machines of either byte order.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cmix {

class Writer {
 public:
  void u8(uint8_t v) { put(&v, 1); }
  void u16(uint16_t v) { uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)}; put(b, 2); }
  void u32(uint32_t v) { for (int i = 0; i < 4; ++i) u8((uint8_t)(v >> (8 * i))); }
  void u64(uint64_t v) { for (int i = 0; i < 8; ++i) u8((uint8_t)(v >> (8 * i))); }
  void i32(int32_t v) { u32((uint32_t)v); }
  void f32(float v) { uint32_t u; std::memcpy(&u, &v, 4); u32(u); }
  void f64(double v) { uint64_t u; std::memcpy(&u, &v, 8); u64(u); }
  void bytes(const void* p, size_t n) { put(p, n); }
  void tag(const char* four) { put(four, 4); }
  void vec_u8(const std::vector<uint8_t>& v) { u64(v.size()); put(v.data(), v.size()); }
  void vec_u16(const std::vector<uint16_t>& v) { u64(v.size()); for (uint16_t x : v) u16(x); }
  void vec_i32(const std::vector<int32_t>& v) { u64(v.size()); for (int32_t x : v) i32(x); }
  void vec_f32(const std::vector<float>& v) { u64(v.size()); for (float x : v) f32(x); }
  const std::vector<uint8_t>& data() const { return buf_; }
  uint64_t checksum() const { return sum_; }

 private:
  void put(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) sum_ = (sum_ ^ b[i]) * 0x100000001b3ull;
    buf_.insert(buf_.end(), b, b + n);
  }
  std::vector<uint8_t> buf_;
  uint64_t sum_ = 0xcbf29ce484222325ull;
};

class Reader {
 public:
  Reader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
  uint8_t u8() { uint8_t v; get(&v, 1); return v; }
  uint16_t u16() { uint8_t b[2]; get(b, 2); return (uint16_t)(b[0] | (b[1] << 8)); }
  uint32_t u32() { uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= (uint32_t)u8() << (8 * i); return v; }
  uint64_t u64() { uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)u8() << (8 * i); return v; }
  int32_t i32() { return (int32_t)u32(); }
  float f32() { uint32_t u = u32(); float v; std::memcpy(&v, &u, 4); return v; }
  double f64() { uint64_t u = u64(); double v; std::memcpy(&v, &u, 8); return v; }
  void bytes(void* p, size_t n) { get(p, n); }
  void expect_tag(const char* four) {
    char t[4];
    get(t, 4);
    if (std::memcmp(t, four, 4) != 0)
      throw std::runtime_error(std::string("state file: expected section ") + std::string(four, 4));
  }
  void vec_u8(std::vector<uint8_t>& v) { v.resize(count(1)); get(v.data(), v.size()); }
  void vec_u16(std::vector<uint16_t>& v) { v.resize(count(2)); for (auto& x : v) x = u16(); }
  void vec_i32(std::vector<int32_t>& v) { v.resize(count(4)); for (auto& x : v) x = i32(); }
  void vec_f32(std::vector<float>& v) { v.resize(count(4)); for (auto& x : v) x = f32(); }
  // Vector read that must match an expected length (table sizes are fixed by config).
  void vec_u16_exact(std::vector<uint16_t>& v, size_t n) {
    vec_u16(v);
    if (v.size() != n) throw std::runtime_error("state file: table size mismatch");
  }
  size_t remaining() const { return n_ - pos_; }
  uint64_t checksum() const { return sum_; }

 private:
  size_t count(size_t elem) {
    uint64_t n = u64();
    if (n > remaining() / elem) throw std::runtime_error("state file: truncated or corrupt");
    return (size_t)n;
  }
  void get(void* p, size_t n) {
    if (n > remaining()) throw std::runtime_error("state file: truncated");
    std::memcpy(p, p_ + pos_, n);
    for (size_t i = 0; i < n; ++i) sum_ = (sum_ ^ p_[pos_ + i]) * 0x100000001b3ull;
    pos_ += n;
  }
  const uint8_t* p_;
  size_t n_;
  size_t pos_ = 0;
  uint64_t sum_ = 0xcbf29ce484222325ull;
};

}  // namespace cmix
