// compress / decompress.
//
// Format "CMX6":
//   magic "CMX6" | type u8 | width u32 | channels u8 | sample rate u32 | LSTM cells u8
//   | graph u8 | graph confirm u32 | snn u8 | snn leak f32 (little-endian)
//   | container header length u32 | container header bytes (PNM / WAV header, kept as is)
//   | payload length u64 | coded payload
// The model learns and codes only the payload (text, pixels, samples).
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "cli/args.h"
#include "cli/commands.h"
#include "cli/common.h"
#include "coder/arith.h"
#include "core/serial.h"
#include "io/files.h"
#include "io/formats.h"
#include "model/session.h"

namespace cmix {

int cmd_compress(int argc, char** argv, int start) {
  Args a(argc, argv, start, {"type", "width", "channels", "lstm", "graph-confirm", "snn-leak"}, {"graph", "snn"});
  if (a.pos().size() != 2)
    throw std::runtime_error(
        "usage: compress [--type text|image|audio|raw] [--width W --channels C] [--lstm N] [--graph [--snn]] <in> <out.cmxb>");
  Config cfg = config_from_args(a);
  const TypedData d = read_typed(a.pos()[0], cfg.type, true);
  apply_shape(cfg, d, true, a.pos()[0]);
  cfg.sample_rate = d.sample_rate;

  Writer w;
  w.tag("CMX6");
  w.u8((uint8_t)cfg.type);
  w.u32((uint32_t)cfg.width);
  w.u8((uint8_t)cfg.channels);
  w.u32((uint32_t)cfg.sample_rate);
  w.u8((uint8_t)cfg.lstm_cells);
  w.u8(cfg.graph ? 1 : 0);
  w.u32((uint32_t)cfg.graph_confirm);
  w.u8(cfg.snn ? 1 : 0);
  w.f32(cfg.snn_leak);
  w.u32((uint32_t)d.header.size());
  w.bytes(d.header.data(), d.header.size());
  w.u64(d.payload.size());
  std::vector<uint8_t> out = w.data();

  Model m(cfg);
  Session s(m);
  s.begin_record();
  Encoder enc(out);
  for (uint8_t b : d.payload) {
    for (int k = 7; k >= 0; --k) {
      int bit = (b >> k) & 1;
      enc.encode(bit, s.predict());
      s.learn_bit(bit);
    }
  }
  enc.flush();
  write_file_atomic(a.pos()[1], out);
  std::cerr << "in  " << d.header.size() + d.payload.size() << " bytes\n";
  std::cerr << "out " << out.size() << " bytes (header+payload)\n";
  return 0;
}

int cmd_decompress(int argc, char** argv, int start) {
  Args a(argc, argv, start, {}, {});
  if (a.pos().size() != 2) throw std::runtime_error("usage: decompress <in.cmxb> <out>");
  const std::string& path = a.pos()[0];
  std::vector<uint8_t> in = read_file(path);
  for (const char* old : {"CMXB", "CMX2", "CMX3", "CMX4", "CMX5"})
    if (in.size() >= 4 && std::memcmp(in.data(), old, 4) == 0)
      throw std::runtime_error(path + " was made by an older version of cmix-bit and cannot be read");
  if (in.size() < 4 || std::memcmp(in.data(), "CMX6", 4) != 0) throw std::runtime_error(path + ": not a CMX6 file");

  Config cfg;
  std::vector<uint8_t> out;
  uint64_t n = 0;
  size_t payload_at = 0;
  try {
    Reader r(in.data(), in.size());
    r.expect_tag("CMX6");
    const uint8_t t = r.u8();
    if (t > (uint8_t)DataType::Raw) throw std::runtime_error("unknown data type");
    cfg.type = (DataType)t;
    cfg.width = (int)r.u32();
    cfg.channels = r.u8();
    cfg.sample_rate = (int)r.u32();
    cfg.lstm_cells = r.u8();
    if (cfg.lstm_cells > LstmState::kMaxCells) throw std::runtime_error("bad LSTM size");
    cfg.graph = r.u8() != 0;
    cfg.graph_confirm = (int)r.u32();
    cfg.snn = r.u8() != 0;
    cfg.snn_leak = r.f32();
    const uint32_t hlen = r.u32();
    if (hlen > r.remaining()) throw std::runtime_error("truncated");
    out.resize(hlen);
    r.bytes(out.data(), hlen);
    n = r.u64();
    payload_at = in.size() - r.remaining();
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(path + ": " + e.what());
  }

  Model m(cfg);
  Session s(m);
  s.begin_record();
  Decoder dec(in.data() + payload_at, in.size() - payload_at);
  out.reserve(out.size() + (size_t)std::min<uint64_t>(n, 1u << 30));
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
  std::cerr << "wrote " << out.size() << " bytes\n";
  return 0;
}

}  // namespace cmix
