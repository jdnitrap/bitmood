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
  if (s == "image") return DataType::Image;
  if (s == "audio") return DataType::Audio;
  if (s == "raw") return DataType::Raw;
  throw std::runtime_error("unknown --type '" + s + "' (use text, image, audio or raw)");
}

std::vector<View> Config::views() const {
  if (!grid) return {};
  switch (type) {
    case DataType::Text:  // rows cut at newlines, plus two widths the data suggests
      return {{ViewKind::Line, 0}, {ViewKind::Auto, 0}, {ViewKind::Auto, 1}};
    case DataType::Image:  // the image row, one pixel, and a width the data suggests
      return {{ViewKind::Fixed, row_bytes()}, {ViewKind::Word, channels}, {ViewKind::Auto, 0}};
    case DataType::Audio:  // one 16-bit sample, one frame (all channels), a suggested width
      return {{ViewKind::Word, 2}, {ViewKind::Word, 2 * channels}, {ViewKind::Auto, 0}};
    case DataType::Raw:  // everything: lines, two suggested widths, 16/32/64-bit words
      return {{ViewKind::Line, 0}, {ViewKind::Auto, 0}, {ViewKind::Auto, 1},
              {ViewKind::Word, 2}, {ViewKind::Word, 4}, {ViewKind::Word, 8}};
  }
  return {};
}

}  // namespace cmix
