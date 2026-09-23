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
  bool grid = true;  // grid views on (off only for A/B comparison)

  // The grid views this type starts with.
  std::vector<View> views() const;
};

}  // namespace cmix
