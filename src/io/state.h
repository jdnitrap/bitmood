// Saved memory: a Model plus the Stream position it stopped at.
//
// Layout: "CMXS" | version | CONF | STRM | model sections | "END!" | checksum
// The checksum (FNV-1a 64 over everything before it) catches truncated or
// damaged files; the version and config catch files from another build or type.
#pragma once

#include <memory>
#include <string>

#include "model/model.h"

namespace cmix {

constexpr uint32_t kStateVersion = 9;

void save_state(const std::string& path, const Model& m, const Stream& s);
// Throws std::runtime_error explaining what is wrong with the file.
std::unique_ptr<Model> load_state(const std::string& path, Stream& s);

}  // namespace cmix
