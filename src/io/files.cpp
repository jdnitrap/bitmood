#include "io/files.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace cmix {

bool file_exists(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return (bool)f;
}

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), {});
}

void write_file_atomic(const std::string& path, const std::vector<uint8_t>& data) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + tmp);
    out.write((const char*)data.data(), (std::streamsize)data.size());
    out.flush();
    if (!out) throw std::runtime_error("write failed: " + tmp);
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::remove(tmp.c_str());
    throw std::runtime_error("cannot replace " + path);
  }
}

}  // namespace cmix
