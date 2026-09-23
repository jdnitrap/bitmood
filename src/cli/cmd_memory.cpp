// Commands that work with saved memory: train, generate, info.
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>

#include "cli/args.h"
#include "cli/commands.h"
#include "core/math.h"
#include "io/files.h"
#include "io/state.h"
#include "model/session.h"

namespace cmix {

namespace {

// Loads the memory at `path` if it exists, otherwise starts a new one of `type`.
std::unique_ptr<Model> open_or_create(const std::string& path, const Args& a, Stream& s, bool& created) {
  created = !file_exists(path);
  if (created) {
    Config cfg;
    cfg.type = parse_type(a.str("type", "text"));
    s = Stream();
    return std::make_unique<Model>(cfg);
  }
  auto m = load_state(path, s);
  if (a.has("type") && parse_type(a.str("type")) != m->config().type)
    throw std::runtime_error(path + " is a " + type_name(m->config().type) + " memory, not " + a.str("type"));
  return m;
}

double bpb(double bits, uint64_t bytes) { return bytes ? bits / (double)bytes : 0.0; }

// Temperature in stretch space: T < 1 sharpens, T > 1 flattens, T = 1 unchanged.
int apply_temperature(int p, double temp) {
  if (temp == 1.0) return p;
  return squash((int)std::lround(stretch(p) / temp));
}

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

int cmd_generate(int argc, char** argv, int start) {
  Args a(argc, argv, start, {"state", "temp", "seed"}, {});
  if (a.pos().empty() || a.pos().size() > 2)
    throw std::runtime_error("usage: generate <nbytes> [prompt] [--state memory.bin] [--temp T] [--seed N]");
  const long long n = std::stoll(a.pos()[0]);
  const std::string prompt = a.pos().size() == 2 ? a.pos()[1] : "The quick brown fox ";
  const double temp = a.num("temp", 1.0);
  if (!(temp > 0)) throw std::runtime_error("--temp must be > 0");

  Stream st;
  std::unique_ptr<Model> m;
  const bool has_memory = a.has("state");
  if (has_memory) m = load_state(a.str("state"), st);
  else m = std::make_unique<Model>(Config());
  Session s(*m, st);

  // With a memory the prompt only sets the context; without one it is all
  // the model has, so it learns from it.
  for (unsigned char ch : prompt) {
    if (has_memory) s.feed_byte(ch);
    else s.learn_byte(ch);
  }

  std::mt19937 rng((uint32_t)a.integer("seed", 0xC0FFEE));
  std::string out = prompt;
  for (long long i = 0; i < n; ++i) {
    int b = 0;
    for (int k = 0; k < 8; ++k) {
      int p = apply_temperature(s.predict(), temp);
      int bit = (int)(rng() & 4095) < p ? 1 : 0;
      s.skip_bit(bit);
      b = (b << 1) | bit;
    }
    out.push_back((char)b);
  }
  std::cout << out << "\n";
  return 0;
}

int cmd_info(int argc, char** argv, int start) {
  Args a(argc, argv, start, {}, {});
  if (a.pos().size() != 1) throw std::runtime_error("usage: info <memory.bin>");
  Stream st;
  auto m = load_state(a.pos()[0], st);
  std::printf("type           %s\n", type_name(m->config().type));
  std::printf("bytes learned  %llu\n", (unsigned long long)m->bytes_learned);
  std::printf("bits/byte      %.3f (average while learning)\n", bpb(m->bits_spent, m->bytes_learned));
  std::printf("history        %zu bytes kept\n", m->history().size());
  std::printf("mixer weights ");
  for (int i = 0; i < m->mixer().size(); ++i) std::printf(" %s=%d", Model::input_name(i), m->mixer().weight(i));
  std::printf("\n");
  return 0;
}

}  // namespace cmix
