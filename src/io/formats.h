// File containers for typed data. The model learns only the payload
// (pixels, samples); the container header is kept aside.
//
//   text, raw - the whole file is payload
//   image     - binary PGM (P5, grey) or PPM (P6, RGB), 8-bit; or raw pixels
//   audio     - WAV, 16-bit PCM
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model/config.h"

namespace cmix {

struct TypedData {
  std::vector<uint8_t> header;   // container bytes before the payload
  std::vector<uint8_t> payload;  // what the model learns
  int width = 0;                 // image: pixels per row
  int height = 0;                // image
  int channels = 0;              // image: 1 or 3; audio: 1 or 2
  int sample_rate = 0;           // audio
};

// Splits a file into header and payload for data type t. Raw pixel files
// (not PNM) are accepted for images when `raw_image_ok` is set.
TypedData read_typed(const std::string& path, DataType t, bool raw_image_ok = false);

// Fills the shape fields of cfg from d (new memory), or checks that they
// match (existing memory). Throws with an explanation on a mismatch.
void apply_shape(Config& cfg, const TypedData& d, bool creating, const std::string& what);

// A playable/viewable file around generated payload bytes.
std::vector<uint8_t> make_container(const Config& cfg, const std::vector<uint8_t>& payload);

}  // namespace cmix
