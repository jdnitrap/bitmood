// What kind of data a model is for, and the settings that shape it.
// The config is stored in every state file so a memory can't be loaded
// into a model of the wrong kind.
#pragma once

#include <cstdint>
#include <string>

namespace cmix {

enum class DataType : uint8_t { Text = 0, Image = 1, Audio = 2, Raw = 3 };

const char* type_name(DataType t);
// Parses "text", "image", "audio", "raw". Throws on anything else.
DataType parse_type(const std::string& s);

struct Config {
  DataType type = DataType::Text;
};

}  // namespace cmix
