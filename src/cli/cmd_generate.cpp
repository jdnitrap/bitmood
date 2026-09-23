// generate: write new bytes from one or more memories.
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "cli/commands.h"
#include "cli/common.h"

namespace cmix {

int cmd_generate(int argc, char** argv, int start) {
  Args a(argc, argv, start, {"state", "blend", "temp", "top-p", "top-k", "seed", "charset", "novelty"}, {"stats"});
  if (a.pos().empty() || a.pos().size() > 2)
    throw std::runtime_error(
        "usage: generate <nbytes> [prompt] [--state memory.bin]... [--blend w1,w2,..] [--temp T]\n"
        "       [--top-p P] [--top-k K] [--seed N] [--charset seen|utf8|ascii|any] [--novelty N] [--stats]");
  const long long n = std::stoll(a.pos()[0]);
  if (n < 0) throw std::runtime_error("nbytes must be >= 0");
  const std::string prompt = a.pos().size() == 2 ? a.pos()[1] : "The quick brown fox ";
  const bool stats = a.has("stats");

  GenSetup setup = load_sources(a, prompt, stats || a.has("novelty"));
  Generator gen(setup.sources, gen_options(a));
  add_standard_constraints(gen, a, setup);
  // Constraints must see the prompt too (e.g. a prompt ending mid UTF-8 sequence).
  // The models already have it, so only constraints are fed here.
  gen.feed_constraints(std::vector<uint8_t>(prompt.begin(), prompt.end()));

  for (long long i = 0; i < n || (gen.must_continue() && i < n + 4); ++i) gen.next();

  const auto& out = gen.output();
  std::cout << prompt;
  std::cout.write((const char*)out.data(), (std::streamsize)out.size());
  std::cout << "\n";

  if (stats) {
    CopyMeter meter(setup.training);
    for (uint8_t b : out) meter.accept(b);
    std::fprintf(stderr, "generated %zu bytes | model surprise %.3f bits/byte | longest copy from training text: %zu bytes\n",
                 out.size(), gen.bits_per_byte(), meter.longest());
  }
  return 0;
}

}  // namespace cmix
