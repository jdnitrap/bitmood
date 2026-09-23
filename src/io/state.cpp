#include "io/state.h"

#include <stdexcept>

#include "core/serial.h"
#include "io/files.h"

namespace cmix {

namespace {

void save_config(Writer& w, const Config& c) {
  w.tag("CONF");
  w.u8((uint8_t)c.type);
  w.u8(c.grid ? 1 : 0);
  w.u8((uint8_t)c.table_bits);
  w.u8((uint8_t)c.history_bits);
  w.u32((uint32_t)c.width);
  w.u8((uint8_t)c.channels);
  w.u32((uint32_t)c.sample_rate);
  w.u8((uint8_t)c.lstm_cells);
  w.u8(c.graph ? 1 : 0);
  w.u32((uint32_t)c.graph_confirm);
}

Config load_config(Reader& r) {
  r.expect_tag("CONF");
  Config c;
  uint8_t t = r.u8();
  if (t > (uint8_t)DataType::Raw) throw std::runtime_error("state file: unknown data type");
  c.type = (DataType)t;
  c.grid = r.u8() != 0;
  c.table_bits = r.u8();
  c.history_bits = r.u8();
  c.width = (int)r.u32();
  c.channels = r.u8();
  c.sample_rate = (int)r.u32();
  c.lstm_cells = r.u8();
  if (c.lstm_cells > LstmState::kMaxCells) throw std::runtime_error("state file: bad LSTM size");
  c.graph = r.u8() != 0;
  c.graph_confirm = (int)r.u32();
  if (c.table_bits < 16 || c.table_bits > 30 || c.history_bits < 16 || c.history_bits > 32)
    throw std::runtime_error("state file: bad table or history size");
  return c;
}

void save_stream(Writer& w, const Stream& s) {
  w.tag("STRM");
  w.i32(s.last_byte);
  for (uint64_t h : s.order_hash) w.u64(h);
  w.u64(s.bytes);
  w.u64(s.word);
  w.u64(s.prev_word);
  w.u32(s.classes);
  w.u64(s.match_ptr);
  w.u32(s.match_len);
  w.u64(s.record_start);
  const LstmState& l = s.lstm;
  for (const float* a : {l.h, l.c, l.h_prev, l.c_prev, l.gi, l.gf, l.go, l.gg})
    for (int i = 0; i < LstmState::kMaxCells; ++i) w.f32(a[i]);
  w.i32(l.x);
  for (float t : l.tree) w.f32(t);
  w.u8(l.ready ? 1 : 0);
  const GraphState& g = s.graph;
  for (uint32_t t : {g.t0, g.t1, g.learn_t1, g.learn_t0, g.learn_next, g.plan, g.expect}) w.u32(t);
  w.u8(g.completed);
  w.u8(g.planned);
  w.bytes(g.word, sizeof g.word);
  w.i32(g.wlen);
  w.u8(g.overflow);
  w.u8(g.capital);
  w.bytes(g.done_word, sizeof g.done_word);
  w.i32(g.done_len);
  for (float t : g.tree) w.f32(t);
  w.u8(g.tree_ready);
  w.u32(g.n);
  w.u32(g.zero_crossings);
  w.i32(g.last_sign);
  w.f64(g.sum_sq);
  w.f64(g.prev_rms);
  w.u64(s.line_start);
  w.u64(s.prev_line_start);
  s.widths.save(w);
}

Stream load_stream(Reader& r) {
  r.expect_tag("STRM");
  Stream s;
  s.last_byte = r.i32();
  for (uint64_t& h : s.order_hash) h = r.u64();
  s.bytes = r.u64();
  s.word = r.u64();
  s.prev_word = r.u64();
  s.classes = r.u32();
  s.match_ptr = r.u64();
  s.match_len = r.u32();
  s.record_start = r.u64();
  LstmState& l = s.lstm;
  for (float* a : {l.h, l.c, l.h_prev, l.c_prev, l.gi, l.gf, l.go, l.gg})
    for (int i = 0; i < LstmState::kMaxCells; ++i) a[i] = r.f32();
  l.x = r.i32();
  for (float& t : l.tree) t = r.f32();
  l.ready = r.u8() != 0;
  GraphState& g = s.graph;
  for (uint32_t* t : {&g.t0, &g.t1, &g.learn_t1, &g.learn_t0, &g.learn_next, &g.plan, &g.expect}) *t = r.u32();
  g.completed = r.u8() != 0;
  g.planned = r.u8() != 0;
  r.bytes(g.word, sizeof g.word);
  g.wlen = r.i32();
  g.overflow = r.u8() != 0;
  g.capital = r.u8() != 0;
  r.bytes(g.done_word, sizeof g.done_word);
  g.done_len = r.i32();
  if (g.wlen < 0 || g.wlen > GraphState::kMaxWord || g.done_len < 0 || g.done_len > GraphState::kMaxWord)
    throw std::runtime_error("state file: bad graph state");
  for (float& t : g.tree) t = r.f32();
  g.tree_ready = r.u8() != 0;
  g.n = r.u32();
  g.zero_crossings = r.u32();
  g.last_sign = r.i32();
  g.sum_sq = r.f64();
  g.prev_rms = r.f64();
  s.line_start = r.u64();
  s.prev_line_start = r.u64();
  s.widths.load(r);
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
