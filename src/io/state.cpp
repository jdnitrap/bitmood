#include "io/state.h"

#include <stdexcept>

#include "core/serial.h"
#include "io/files.h"

namespace cmix {

namespace {

void save_config(Writer& w, const Config& c) {
  w.tag("CONF");
  w.u8((uint8_t)c.type);
}

Config load_config(Reader& r) {
  r.expect_tag("CONF");
  Config c;
  uint8_t t = r.u8();
  if (t > (uint8_t)DataType::Raw) throw std::runtime_error("state file: unknown data type");
  c.type = (DataType)t;
  return c;
}

void save_stream(Writer& w, const Stream& s) {
  w.tag("STRM");
  w.i32(s.last_byte);
  for (uint32_t h : s.order_hash) w.u32(h);
  w.u64(s.bytes);
}

Stream load_stream(Reader& r) {
  r.expect_tag("STRM");
  Stream s;
  s.last_byte = r.i32();
  for (uint32_t& h : s.order_hash) h = r.u32();
  s.bytes = r.u64();
  return s;
}

}  // namespace

void save_state(const std::string& path, const Model& m, const Stream& s) {
  Writer w;
  w.tag("CMXS");
  w.u32(kStateVersion);
  save_config(w, m.config());
  save_stream(w, s);
  m.save(w);
  w.tag("END!");
  std::vector<uint8_t> out = w.data();
  uint64_t sum = w.checksum();
  for (int i = 0; i < 8; ++i) out.push_back((uint8_t)(sum >> (8 * i)));
  write_file_atomic(path, out);
}

std::unique_ptr<Model> load_state(const std::string& path, Stream& s) {
  std::vector<uint8_t> data = read_file(path);
  if (data.size() < 16) throw std::runtime_error(path + ": too small to be a state file");
  try {
    Reader r(data.data(), data.size() - 8);
    r.expect_tag("CMXS");
    uint32_t ver = r.u32();
    if (ver != kStateVersion)
      throw std::runtime_error("state file version " + std::to_string(ver) + ", this build reads " +
                               std::to_string(kStateVersion) + "; retrain with this build");
    Config cfg = load_config(r);
    s = load_stream(r);
    auto m = std::make_unique<Model>(cfg);
    m->load(r);
    r.expect_tag("END!");
    uint64_t want = 0;
    for (int i = 0; i < 8; ++i) want |= (uint64_t)data[data.size() - 8 + i] << (8 * i);
    if (r.remaining() != 0 || r.checksum() != want)
      throw std::runtime_error("checksum mismatch (file damaged)");
    return m;
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
}

}  // namespace cmix
