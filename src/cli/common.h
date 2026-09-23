// Helpers shared by several commands.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cli/args.h"
#include "gen/generator.h"
#include "gen/suffix_array.h"
#include "model/model.h"

namespace cmix {

inline double per_byte(double bits, uint64_t n) { return n ? bits / (double)n : 0.0; }

// Loads the memory at `path` if it exists, otherwise starts a new one using
// the type options in `a` (--type ...). Checks --type against a loaded memory.
std::unique_ptr<Model> open_or_create(const std::string& path, const Args& a, Stream& s, bool& created);

// The models a generation command works from, plus the text they were
// trained on (for the novelty cap and copy statistics).
struct GenSetup {
  std::vector<std::unique_ptr<Model>> models;
  std::vector<Source> sources;
  std::shared_ptr<const SuffixArray> training;  // null unless needed
};

// Reads --state (repeatable), --blend, and builds sources. With no --state,
// one fresh model learns the prompt. `need_training_text` builds the suffix
// array over every memory's history before the prompt is added.
GenSetup load_sources(const Args& a, const std::string& prompt, bool need_training_text);

// --temp, --top-p, --top-k, --seed.
GenOptions gen_options(const Args& a);

// --charset (default seen) and --novelty N.
void add_standard_constraints(Generator& g, const Args& a, const GenSetup& setup);

}  // namespace cmix
