// Commands that work with saved memory: train, info.
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "cli/args.h"
#include "cli/common.h"
#include "cli/commands.h"
#include "io/files.h"
#include "io/formats.h"
#include "io/state.h"
#include "model/session.h"

namespace cmix {

namespace {

double bpb(double bits, uint64_t bytes) { return bytes ? bits / (double)bytes : 0.0; }

}  // namespace

int cmd_train(int argc, char** argv, int start) {
  Args a(argc, argv, start, {"state", "type", "table-bits", "width", "channels", "lstm", "graph-confirm"}, {"graph"});
  if (!a.has("state") || a.pos().empty())
    throw std::runtime_error(
        "usage: train --state <memory.bin> [--type text|image|audio|raw] [--table-bits 16..28] [--lstm N]\n"
        "             [--graph [--graph-confirm N]]\n"
        "             [--width W --channels 1|3 (raw pixel files)] <file>...");
  const std::string path = a.str("state");
  // The data type decides how files are read; a new image/audio memory
  // takes its shape from the first file.
  const DataType type = file_exists(path) ? DataType::Text : parse_type(a.str("type", "text"));
  Stream st;
  bool created = false;
  std::unique_ptr<Model> m;
  if (file_exists(path)) {
    m = open_or_create(path, a, st, created);
  } else {
    const TypedData first = read_typed(a.pos()[0], type, true);
    m = open_or_create(path, a, st, created, &first);
  }
  std::cerr << (created ? "new " : "loaded ") << type_name(m->config().type) << " memory " << path;
  if (!created) std::cerr << " (" << m->bytes_learned << " bytes learned so far)";
  std::cerr << "\n";

  Session s(*m, st);
  const Config& cfg = m->config();
  const bool records = cfg.type == DataType::Image || cfg.type == DataType::Audio;
  for (const std::string& f : a.pos()) {
    const TypedData d = read_typed(f, cfg.type, true);
    Config check = cfg;
    apply_shape(check, d, false, f);
    if (records) s.begin_record();  // each image / sound starts at pixel / sample 0
    double bits = 0;
    for (uint8_t b : d.payload) bits += s.learn_byte(b);
    std::fprintf(stderr, "  %-40s %10zu bytes  %6.3f bits/byte\n", f.c_str(), d.payload.size(),
                 bpb(bits, d.payload.size()));
  }
  save_state(path, *m, s.stream());
  std::fprintf(stderr, "saved %s: %llu bytes learned in total, %.3f bits/byte average\n", path.c_str(),
               (unsigned long long)m->bytes_learned, bpb(m->bits_spent, m->bytes_learned));
  return 0;
}

int cmd_info(int argc, char** argv, int start) {
  Args a(argc, argv, start, {}, {});
  if (a.pos().size() != 1) throw std::runtime_error("usage: info <memory.bin>");
  Stream st;
  auto m = load_state(a.pos()[0], st);
  std::printf("type           %s", type_name(m->config().type));
  if (m->config().type == DataType::Image)
    std::printf(" (%d px wide, %d channel%s)", m->config().width, m->config().channels, m->config().channels > 1 ? "s" : "");
  if (m->config().type == DataType::Audio)
    std::printf(" (%d channel%s, %d Hz)", m->config().channels, m->config().channels > 1 ? "s" : "", m->config().sample_rate);
  std::printf("\n");
  std::printf("bytes learned  %llu\n", (unsigned long long)m->bytes_learned);
  std::printf("bits/byte      %.3f average while learning, %.3f recently\n", bpb(m->bits_spent, m->bytes_learned),
              m->recent_bpb);
  std::printf("history        %zu bytes kept\n", m->history().size());
  std::printf("tables         2^%d slots (%d MB), history up to %d MB\n", m->config().table_bits,
              (int)((8ull << m->config().table_bits) >> 20), (int)((1ull << m->config().history_bits) >> 20));
  if (m->config().lstm_cells) std::printf("LSTM           %d cells\n", m->config().lstm_cells);
  if (const TokenGraph* g = m->graph())
    std::printf("graph          %zu nodes, %zu edges (%zu confirmed, needs %u sightings)%s\n", g->nodes(),
                g->edges(false), g->edges(true), g->confirm(),
                m->config().type == DataType::Text ? (", " + std::to_string(m->vocab().size()) + " words").c_str() : "");
  std::printf("specialists   ");
  for (int i = 0; i < m->num_inputs(); ++i) std::printf(" %s", m->input_name(i).c_str());
  std::printf("\n");
  std::printf("grid views    ");
  for (int v = 0; v < m->grid().num_views(); ++v) std::printf(" [%s]", view_label(m->grid().view(v), st).c_str());
  std::printf("\n");
  return 0;
}

}  // namespace cmix
