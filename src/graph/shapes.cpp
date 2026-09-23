#include "graph/shapes.h"

#include <cmath>
#include <cstdlib>

namespace cmix {

// ---- sound ------------------------------------------------------------------

Token audio_code(double rms, uint32_t zero_crossings, double prev_rms) {
  const int loud = std::min(7, (int)(std::log2(rms + 1.0) / 2.0));
  const int pitch = std::min(7, (int)std::log2((double)zero_crossings + 1.0));
  const int trend = rms > prev_rms * 1.25 + 1 ? 2 : (rms < prev_rms * 0.8 ? 0 : 1);
  return (Token)((loud * 8 + pitch) * 3 + trend);
}

std::string describe_audio(Token code, int sample_rate, int slice_samples) {
  if (code >= (Token)kAudioCodes) return "?";
  const int trend = (int)(code % 3), pitch = (int)(code / 3 % 8), loud = (int)(code / 24);
  // Zero crossings in the slice ~ 2^pitch; two crossings per wave cycle.
  const double hz = slice_samples > 0 ? (std::pow(2.0, pitch + 0.5) / 2.0) * sample_rate / slice_samples : 0;
  static const char* trends[3] = {"fading", "steady", "rising"};
  return "loud " + std::to_string(loud) + "/7, ~" + std::to_string((int)hz) + " Hz, " + trends[trend];
}

// ---- pictures ---------------------------------------------------------------

Token image_run_code(const History& h, uint64_t record_start, int width, int channels, uint64_t y, int x0, int x1) {
  if (x1 <= x0) return kNoToken;
  const uint64_t row = (uint64_t)width * (uint64_t)channels;
  const uint64_t first = record_start + y * row + (uint64_t)x0 * (uint64_t)channels;
  const uint64_t last = record_start + y * row + (uint64_t)x1 * (uint64_t)channels - 1;
  if (!h.has(first) || !h.has(last)) return kNoToken;
  int lum[kRunPixels];
  long ch_sum[3] = {0, 0, 0};
  const int n = x1 - x0;
  for (int i = 0; i < n; ++i) {
    const uint64_t p = first + (uint64_t)i * (uint64_t)channels;
    int sum = 0;
    for (int c = 0; c < channels; ++c) {
      const int v = h.at(p + (uint64_t)c);
      sum += v;
      if (c < 3) ch_sum[c] += v;
    }
    lum[i] = sum / channels;
  }
  long total = 0;
  for (int i = 0; i < n; ++i) total += lum[i];
  const int mean = (int)(total / n);
  long a = 0, b = 0;
  for (int i = 0; i < n; ++i) (i < n / 2 ? a : b) += lum[i];
  const int half = std::max(1, n / 2);
  const int diff = (int)(b / std::max(1, n - half) - a / half);
  const int slope = diff > 12 ? 2 : (diff < -12 ? 0 : 1);
  int busy = 0;
  for (int i = 1; i < n; ++i) busy += std::abs(lum[i] - lum[i - 1]);
  busy /= std::max(1, n - 1);
  const int texture = busy < 4 ? 0 : (busy < 16 ? 1 : 2);
  int colour = 3;
  if (channels >= 3) {
    int hi = 0, lo = 0;
    for (int c = 1; c < 3; ++c) {
      if (ch_sum[c] > ch_sum[hi]) hi = c;
      if (ch_sum[c] < ch_sum[lo]) lo = c;
    }
    if ((ch_sum[hi] - ch_sum[lo]) / n > 24) colour = hi;
  }
  return (Token)((((mean >> 5) * 3 + slope) * 3 + texture) * 4 + colour);
}

std::string describe_image(Token code) {
  if (code >= (Token)kImageCodes) return "?";
  static const char* slopes[3] = {"darker to the right", "flat", "lighter to the right"};
  static const char* textures[3] = {"smooth", "some texture", "busy"};
  static const char* colours[4] = {"reddish", "greenish", "bluish", "neutral"};
  const int colour = (int)(code % 4), texture = (int)(code / 4 % 3), slope = (int)(code / 12 % 3),
            mean = (int)(code / 36);
  return "brightness " + std::to_string(mean) + "/7, " + slopes[slope] + ", " + textures[texture] + ", " +
         colours[colour];
}

}  // namespace cmix
