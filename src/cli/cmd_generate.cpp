// generate: write new bytes from one or more memories.
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "cli/commands.h"
#include "cli/common.h"
#include "gen/best_of.h"
#include "io/files.h"
#include "io/formats.h"

namespace cmix {

int cmd_generate(int argc, char** argv, int start) {
  Args a(argc, argv, start, kGenValued, kGenFlags);
  const char* usage =
      "usage: generate <nbytes> [prompt] [--state memory.bin]... [--blend w1,w2,..] [--blend-mode mix|product]\n"
      "       [--temp T] [--top-p P] [--top-k K] [--seed N] [--charset seen|utf8|ascii|any]\n"
      "       [--novelty N] [--line-start CHARS | --acrostic WORD] [--max-line N]\n"
      "       [--words FILE] [--rhyme] [--best-of N] [--graph-plan [--plan-strength S] [--plan-temp T]] [--stats]\n"
      "   image memory: generate --state img.bin --height H --out new.ppm\n"
      "   audio memory: generate --state snd.bin --seconds S --out new.wav\n"
      "   raw memory:   generate <nbytes> --state raw.bin [--out file]";
  if (a.pos().size() > 2) throw std::runtime_error(usage);
  // Text prompt: given, or the classic default when only nbytes is given.
  const std::string prompt = a.pos().size() == 2 ? a.pos()[1] : (a.pos().empty() ? "" : "The quick brown fox ");
  const bool stats = a.has("stats");
  const long long best_of = a.integer("best-of", 1);
  if (best_of < 1) throw std::runtime_error("--best-of must be >= 1");

  GenSetup setup = load_sources(a, prompt, stats || a.has("novelty") || best_of > 1);
  const Config cfg = setup.models[0]->config();
  long long n = a.pos().empty() ? -1 : std::stoll(a.pos()[0]);
  if (cfg.type == DataType::Image && a.has("height")) n = a.integer("height", 0) * cfg.row_bytes();
  if (cfg.type == DataType::Audio && a.has("seconds"))
    n = (long long)(a.num("seconds", 0) * cfg.sample_rate) * 2 * cfg.channels;
  if (n < 0) throw std::runtime_error(usage);
  if ((cfg.type == DataType::Image || cfg.type == DataType::Audio) && !a.has("out"))
    throw std::runtime_error(std::string(type_name(cfg.type)) + " output needs --out FILE");
  const GenOptions opt = gen_options(a);
  Generator gen(setup.sources, opt);
  add_standard_constraints(gen, a, setup);
  add_shape_constraints(gen, a);
  // Constraints must see the prompt too (e.g. a prompt ending mid UTF-8 sequence).
  // The models already have it, so only constraints are fed here.
  gen.feed_constraints(std::vector<uint8_t>(prompt.begin(), prompt.end()));
  gen.begin_output();

  if (best_of > 1) {
    BestOfOptions bo;
    bo.candidates = (int)best_of;
    const double t = gen.training_bits_per_byte();
    if (t > 0) bo.target_bpb = t;
    if (!setup.models.empty() && setup.models[0]->history().size() > 0)
      bo.typical_len = typical_line_length(setup.models[0]->history().data());
    uint64_t unit = 0;
    while ((long long)gen.output().size() < n) {
      Candidate c = best_of_line(gen, bo, setup.training, opt.seed, unit++);
      if (stats)
        std::fprintf(stderr, "line %llu: best of %lld, %.2f bits/byte (target %.2f), %d spikes, copy %zu\n",
                     (unsigned long long)unit, best_of, c.bpb, bo.target_bpb, c.spikes, c.copy);
    }
  } else {
    for (long long i = 0; i < n || (gen.must_continue() && i < n + 256); ++i) gen.next();
  }

  const auto& out = gen.output();
  if (a.has("out")) {
    write_file_atomic(a.str("out"), make_container(cfg, out));
  } else {
    std::cout << prompt;
    std::cout.write((const char*)out.data(), (std::streamsize)out.size());
    if (cfg.type == DataType::Text) std::cout << "\n";
  }

  if (stats) {
    CopyMeter meter(setup.training);
    for (uint8_t b : out) meter.accept(b);
    std::fprintf(stderr, "generated %zu bytes | model surprise %.3f bits/byte | longest copy from training text: %zu bytes\n",
                 out.size(), gen.bits_per_byte(), meter.longest());
  }
  return 0;
}

}  // namespace cmix
