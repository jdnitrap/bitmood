#include "gen/suffix_array.h"

#include <algorithm>
#include <numeric>

namespace cmix {

SuffixArray::SuffixArray(std::vector<uint8_t> text) : text_(std::move(text)) {
  const size_t n = text_.size();
  if (n == 0) return;
  // Prefix doubling with counting sorts on (rank[i], rank[i + k]).
  // Rank 0 is reserved for "past the end", so text ranks are 1..256.
  std::vector<uint32_t> rank(n), tmp(n), sa(n), sa2(n), cnt;
  for (size_t i = 0; i < n; ++i) rank[i] = text_[i] + 1u;
  std::iota(sa.begin(), sa.end(), 0u);
  uint32_t classes = 257;
  for (size_t k = 1;; k <<= 1) {
    auto second = [&](uint32_t i) { return i + k < n ? rank[i + k] : 0u; };
    // Sort by second key, then stable sort by first key.
    cnt.assign(classes + 1, 0);
    for (size_t i = 0; i < n; ++i) ++cnt[second((uint32_t)i)];
    for (size_t c = 1; c < cnt.size(); ++c) cnt[c] += cnt[c - 1];
    for (size_t i = n; i-- > 0;) sa2[--cnt[second((uint32_t)i)]] = (uint32_t)i;
    cnt.assign(classes + 1, 0);
    for (size_t i = 0; i < n; ++i) ++cnt[rank[i]];
    for (size_t c = 1; c < cnt.size(); ++c) cnt[c] += cnt[c - 1];
    for (size_t j = n; j-- > 0;) sa[--cnt[rank[sa2[j]]]] = sa2[j];
    // New ranks.
    tmp[sa[0]] = 1;
    for (size_t j = 1; j < n; ++j) {
      bool same = rank[sa[j]] == rank[sa[j - 1]] && second(sa[j]) == second(sa[j - 1]);
      tmp[sa[j]] = tmp[sa[j - 1]] + (same ? 0 : 1);
    }
    rank.swap(tmp);
    classes = rank[sa[n - 1]];
    if (classes == n) break;
  }
  sa_ = std::move(sa);
}

SuffixArray::Range SuffixArray::extend(Range r, size_t k, uint8_t c) const {
  // Within r all suffixes share the first k bytes; they are sorted by byte k.
  size_t lo = r.lo, hi = r.hi;
  size_t a = lo, b = hi;
  while (a < b) {  // first suffix with byte k >= c
    size_t mid = (a + b) / 2;
    if (at(mid, k) < (int)c) a = mid + 1;
    else b = mid;
  }
  size_t first = a;
  b = hi;
  while (a < b) {  // first suffix with byte k > c
    size_t mid = (a + b) / 2;
    if (at(mid, k) <= (int)c) a = mid + 1;
    else b = mid;
  }
  return {first, a};
}

SuffixArray::Range SuffixArray::find(const uint8_t* pattern, size_t n) const {
  Range r{0, sa_.size()};
  for (size_t k = 0; k < n && !r.empty(); ++k) r = extend(r, k, pattern[k]);
  return r;
}

}  // namespace cmix
