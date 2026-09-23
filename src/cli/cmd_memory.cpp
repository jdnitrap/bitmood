// Commands that work with saved memory: train, info.
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "cli/args.h"
#include "cli/common.h"
#include "cli/commands.h"
#include "io/files.h"
#include "io/state.h"
#include "model/session.h"

namespace cmix {

namespace {

double bpb(double bits, uint64_t bytes) { return bytes ? bits / (double)bytes : 0.0; }

}  // namespace

int cmd_train(int argc, char** argv, int start) {
  Args a(argc, argv, start, {"state", "type"}, {});
  if (!a.has("state") || a.pos().empty())
    throw std::runtime_error("usage: train --state <memory.bin> [--type text] <file>...");
  const std::string path = a.str("state");
  Stream st;
  bool created = false;
  auto m = open_or_create(path, a, st, created);
  std::cerr << (created ? "new " : "loaded ") << type_name(m->config().type) << " memory " << path;
  if (!created) std::cerr << " (" << m->bytes_learned << " bytes learned so far)";
  std::cerr << "\n";

  Session s(*m, st);
  for (const std::string& f : a.pos()) {
    std::vector<uint8_t> data = read_file(f);
    double bits = 0;
    for (uint8_t b : data) bits += s.learn_byte(b);
    std::fprintf(stderr, "  %-40s %10zu bytes  %6.3f bits/byte\n", f.c_str(), data.size(), bpb(bits, data.size()));
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
  std::printf("type           %s\n", type_name(m->config().type));
  std::printf("bytes learned  %llu\n", (unsigned long long)m->bytes_learned);
  std::printf("bits/byte      %.3f average while learning, %.3f recently\n", bpb(m->bits_spent, m->bytes_learned),
              m->recent_bpb);
  std::printf("history        %zu bytes kept\n", m->history().size());
  std::printf("mixer weights ");
  for (int i = 0; i < m->mixer().size(); ++i)
    std::printf(" %s=%d", m->input_name(i).c_str(), m->mixer().weight(i, m->best_view()));
  std::printf("\n");
  std::printf("grid views    ");
  for (int v = 0; v < m->grid().num_views(); ++v) std::printf(" [%s]", view_label(m->grid().view(v), st).c_str());
  std::printf("\n");
  return 0;
}

}  // namespace cmix
