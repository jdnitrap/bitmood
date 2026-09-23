#include "model/config.h"

#include <stdexcept>

#include "model/grid.h"

namespace cmix {

const char* type_name(DataType t) {
  switch (t) {
    case DataType::Text: return "text";
    case DataType::Image: return "image";
    case DataType::Audio: return "audio";
    case DataType::Raw: return "raw";
  }
  return "?";
}

DataType parse_type(const std::string& s) {
  if (s == "text") return DataType::Text;
  if (s == "image" || s == "audio" || s == "raw")
    throw std::runtime_error("--type " + s + " is not implemented yet");
  throw std::runtime_error("unknown --type '" + s + "' (use text, image, audio or raw)");
}

std::vector<View> Config::views() const {
  if (!grid) return {};
  // Text: rows cut at newlines, plus the two row widths the data suggests.
  return {{ViewKind::Line, 0}, {ViewKind::Auto, 0}, {ViewKind::Auto, 1}};
}

}  // namespace cmix
