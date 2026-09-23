// compress / decompress.
// Format: "CMX2" | data type (u8) | original length (u64, big-endian) | payload.
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "cli/args.h"
#include "cli/commands.h"
#include "coder/arith.h"
#include "io/files.h"
#include "model/session.h"

namespace cmix {

namespace {

constexpr size_t kHeader = 13;

void put_u64_be(std::vector<uint8_t>& o, uint64_t v) {
  for (int i = 7; i >= 0; --i) o.push_back((uint8_t)(v >> (i * 8)));
}

uint64_t get_u64_be(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
  return v;
}

}  // namespace

int cmd_compress(int argc, char** argv, int start) {
  Args a(argc, argv, start, {"type"}, {});
  if (a.pos().size() != 2) throw std::runtime_error("usage: compress [--type text] <in> <out.cmxb>");
  Config cfg;
  cfg.type = parse_type(a.str("type", "text"));
  std::vector<uint8_t> data = read_file(a.pos()[0]);
  std::vector<uint8_t> out = {'C', 'M', 'X', '2', (uint8_t)cfg.type};
  put_u64_be(out, data.size());

  Model m(cfg);
  Session s(m);
  Encoder enc(out);
  for (uint8_t b : data) {
    for (int k = 7; k >= 0; --k) {
      int bit = (b >> k) & 1;
      enc.encode(bit, s.predict());
      s.learn_bit(bit);
    }
  }
  enc.flush();
  write_file_atomic(a.pos()[1], out);
  std::cerr << "in  " << data.size() << " bytes\n";
  std::cerr << "out " << out.size() << " bytes (header+payload)\n";
  return 0;
}

int cmd_decompress(int argc, char** argv, int start) {
  Args a(argc, argv, start, {}, {});
  if (a.pos().size() != 2) throw std::runtime_error("usage: decompress <in.cmxb> <out>");
  std::vector<uint8_t> in = read_file(a.pos()[0]);
  if (in.size() >= 4 && std::memcmp(in.data(), "CMXB", 4) == 0)
    throw std::runtime_error(a.pos()[0] + " was made by an older version (before phase 5) and cannot be read");
  if (in.size() < kHeader || std::memcmp(in.data(), "CMX2", 4) != 0)
    throw std::runtime_error(a.pos()[0] + ": not a CMX2 file");
  if (in[4] > (uint8_t)DataType::Raw) throw std::runtime_error(a.pos()[0] + ": unknown data type");
  Config cfg;
  cfg.type = (DataType)in[4];
  const uint64_t n = get_u64_be(in.data() + 5);

  Model m(cfg);
  Session s(m);
  Decoder dec(in.data() + kHeader, in.size() - kHeader);
  std::vector<uint8_t> out;
  out.reserve((size_t)std::min<uint64_t>(n, 1u << 30));
  for (uint64_t i = 0; i < n; ++i) {
    int b = 0;
    for (int k = 0; k < 8; ++k) {
      int bit = dec.decode(s.predict());
      s.learn_bit(bit);
      b = (b << 1) | bit;
    }
    out.push_back((uint8_t)b);
  }
  write_file_atomic(a.pos()[1], out);
  std::cerr << "wrote " << n << " bytes\n";
  return 0;
}

}  // namespace cmix
