// Helpers shared by several commands.
#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cli/args.h"
#include "gen/generator.h"
#include "gen/suffix_array.h"
#include "io/formats.h"
#include "model/model.h"

namespace cmix {

inline double per_byte(double bits, uint64_t n) { return n ? bits / (double)n : 0.0; }

// A new memory's config from --type, --table-bits, --width, --channels.
Config config_from_args(const Args& a);

// Loads the memory at `path` if it exists, otherwise starts a new one using
// the options in `a`. For image and audio memories the shape (width,
// channels, sample rate) comes from `first` (the first training file) when
// creating. Checks --type and --table-bits against a loaded memory.
std::unique_ptr<Model> open_or_create(const std::string& path, const Args& a, Stream& s, bool& created,
                                      const TypedData* first = nullptr);

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

// --blend-mode, --temp, --top-p, --top-k, --seed.
GenOptions gen_options(const Args& a);

// --charset (default seen for text, any otherwise) and --novelty N.
void add_standard_constraints(Generator& g, const Args& a, const GenSetup& setup);

// --line-start CHARS, --acrostic WORD, --max-line N, --words FILE, --rhyme.
void add_shape_constraints(Generator& g, const Args& a);

// Option names used by the commands that generate.
extern const std::set<std::string> kGenValued;
extern const std::set<std::string> kGenFlags;

}  // namespace cmix
