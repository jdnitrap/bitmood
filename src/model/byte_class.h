// Byte classes: the MDBE "named flags" (is_alpha, is_digit, is_upper,
// is_punct, is_space, utf8_lead) folded into one small class number.
// Specialist C learns from the classes of recent bytes instead of using
// hard-coded rules, so it works for any data, not only English text.
#pragma once

#include <cstdint>

namespace cmix {

enum ByteClass : uint8_t {
  kClsStart = 0,  // no byte yet
  kClsLower,
  kClsUpper,
  kClsDigit,
  kClsSpace,      // space, tab
  kClsNewline,    // \n, \r
  kClsPunct,
  kClsUtf8Lead,
  kClsUtf8Cont,
  kClsControl,
  kClsHigh,       // other bytes >= 0x7F
  kNumClasses
};

inline ByteClass byte_class(int b) {
  if (b < 0) return kClsStart;
  if (b >= 'a' && b <= 'z') return kClsLower;
  if (b >= 'A' && b <= 'Z') return kClsUpper;
  if (b >= '0' && b <= '9') return kClsDigit;
  if (b == ' ' || b == '\t') return kClsSpace;
  if (b == '\n' || b == '\r') return kClsNewline;
  if ((b >= 33 && b <= 47) || (b >= 58 && b <= 64) || (b >= 91 && b <= 96) || (b >= 123 && b <= 126))
    return kClsPunct;
  if (b < 32) return kClsControl;
  if ((b & 0xC0) == 0x80) return kClsUtf8Cont;
  if (b >= 0xC2 && b <= 0xF4) return kClsUtf8Lead;
  return kClsHigh;
}

}  // namespace cmix
