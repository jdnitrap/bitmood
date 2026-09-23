// Shape codes: how sound and pictures are cut into graph nodes.
//
// Sound: a 20 ms slice of channel 0 becomes one code from
//   loudness (8 levels, from RMS), pitch-ish (8 levels, from how often the
//   wave crosses zero) and trend (quieter / same / louder than the slice before).
// Pictures: a run of 8 pixels in a row becomes one code from
//   brightness (8 levels), slope (darker / flat / lighter to the right),
//   texture (smooth / some / busy) and dominant colour (red / green / blue / none).
#pragma once

#include <cstdint>
#include <string>

#include "graph/token_graph.h"
#include "model/history.h"

namespace cmix {

// ---- sound ------------------------------------------------------------------

constexpr int kAudioCodes = 8 * 8 * 3;
Token audio_code(double rms, uint32_t zero_crossings, double prev_rms);
std::string describe_audio(Token code, int sample_rate, int slice_samples);

// ---- pictures ---------------------------------------------------------------

constexpr int kRunPixels = 8;
constexpr int kImageCodes = 8 * 3 * 3 * 4;
// Code for pixels [x0, x1) of image row y, read from history. `record_start`
// is the absolute position of the image's first byte. kNoToken if the row
// isn't in history.
Token image_run_code(const History& h, uint64_t record_start, int width, int channels, uint64_t y, int x0, int x1);
std::string describe_image(Token code);

}  // namespace cmix
