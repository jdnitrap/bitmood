// Whole-file read/write helpers.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cmix {

bool file_exists(const std::string& path);
// Throws std::runtime_error with the path on failure.
std::vector<uint8_t> read_file(const std::string& path);
// Writes to path + ".tmp" then renames, so a crash never leaves a half-written file.
void write_file_atomic(const std::string& path, const std::vector<uint8_t>& data);

}  // namespace cmix
