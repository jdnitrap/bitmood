// What kind of data a model is for, and the settings that shape it.
// The config is stored in every state file so a memory can't be loaded
// into a model of the wrong kind.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cmix {

enum class DataType : uint8_t { Text = 0, Image = 1, Audio = 2, Raw = 3 };

const char* type_name(DataType t);
// Parses "text", "image", "audio", "raw". Throws on anything else.
DataType parse_type(const std::string& s);

struct View;

struct Config {
  DataType type = DataType::Text;
  bool grid = true;        // grid views on (off only for A/B comparison)
  int table_bits = 22;     // shared context table: 2^bits slots of 8 bytes
  int history_bits = 24;   // bytes of history kept for long match: 2^bits
  // Shape of typed data (0 = not set).
  int width = 0;           // image: pixels per row
  int channels = 0;        // image: 1 (grey) or 3 (RGB); audio: 1 or 2
  int sample_rate = 0;     // audio (only used when writing WAV files)
  int lstm_cells = 0;      // level 2 LSTM specialist size (0 = off)
  bool graph = false;      // graph specialist G on
  int graph_confirm = 2;   // times an edge must be seen before it votes

  // The grid views this type starts with.
  std::vector<View> views() const;
  // Word specialist (text only).
  bool words() const { return type == DataType::Text; }
  // Bytes per image row.
  int row_bytes() const { return width * channels; }
  // Extra specialists: image (I1..I6) and audio (S1..S4).
  int image_inputs() const { return type == DataType::Image ? 6 : 0; }
  int audio_inputs() const { return type == DataType::Audio ? 4 : 0; }
  // Graph specialist: a table-backed context for audio/image, a direct vote for text.
  int graph_table_inputs() const { return graph && type != DataType::Text && type != DataType::Raw ? 1 : 0; }
  int graph_vote_inputs() const { return graph && type == DataType::Text ? 1 : 0; }
};

}  // namespace cmix
