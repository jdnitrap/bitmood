// Text (or byte) generator on top of one or more models.
//
// For each byte it asks every source model for P(bit = 1) at all 255 nodes
// of the 8-level bit tree, turns that into a probability for each of the
// 256 bytes, blends the sources (see BlendMode), removes bytes the
// constraints forbid, applies temperature / top-k / top-p, and samples.
//
// Models are never taught their own output: the chosen byte only moves each
// source's context forward (Model::advance_byte).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "gen/constraints.h"
#include "model/model.h"

namespace cmix {

struct Source {
  Model* model;
  Stream stream;
  double weight = 1.0;
};

// How several memories are combined.
//   Mix     - weighted average of their byte distributions: each model's
//             confident choices survive (style switching)
//   Product - weighted average in stretch space at every bit: only what
//             they agree on stays likely (style compromise)
enum class BlendMode { Mix, Product };

struct GenOptions {
  BlendMode blend = BlendMode::Mix;
  double temp = 1.0;   // < 1 sharper, > 1 flatter
  double top_p = 1.0;  // keep the smallest set of bytes whose probability sums to top_p
  int top_k = 0;       // keep only the k most likely bytes (0 = off)
  bool plan = false;   // graph planning: pick the next word / slice / run to aim for
  // How hard generation leans toward the plan (see Model::plan_bias).
  // Negative = the default for each memory's type (text 4, audio 100, image 2).
  double plan_strength = -1.0;
  double plan_temp = 1.0;      // temperature for choosing plans (1 = as often as in training)
  uint64_t seed = 0xC0FFEE;
};

using ByteProbs = std::array<double, 256>;

double default_plan_strength(DataType t);

class Generator {
 public:
  Generator(std::vector<Source> sources, GenOptions opt);

  void add_constraint(std::unique_ptr<Constraint> c) { constraints_.push_back(std::move(c)); }
  // Moves the context over a byte without sampling it (the prompt).
  void feed(uint8_t b);
  // Shows bytes to the constraints only (models already have them).
  void feed_constraints(const std::vector<uint8_t>& bytes) {
    for (uint8_t b : bytes)
      for (auto& c : constraints_) c->accept(b);
  }
  void begin_output() {
    for (auto& c : constraints_) c->begin_output();
  }
  // Samples, emits and returns the next byte.
  uint8_t next();
  // Emits a byte chosen elsewhere (replaying a best-of-N winner); `bits` is its model cost.
  void emit(uint8_t b, double bits);
  // Teaches every source model a byte (interactive writing), then moves on.
  void learn_byte(uint8_t b);
  // Frees history room after the models have grown (call between snapshots).
  void ensure_room();
  const Stream& stream(size_t i) const { return src_[i].stream; }
  Model& model(size_t i) { return *src_[i].model; }
  // New random stream (each best-of-N candidate gets its own).
  void reseed(uint64_t seed) { rng_.seed(seed); }
  // True if the constraints say output can't end here (e.g. mid UTF-8).
  bool must_continue() const;
  // Sum of the constraints' satisfaction() (best-of-N scoring).
  double satisfaction() const;

  // Blended model probabilities for the next byte, before any constraint.
  void distribution(ByteProbs& p) const;
  // Bytes allowed next under the constraints (soft ones relaxed if needed).
  ByteMask allowed() const;

  const std::vector<uint8_t>& output() const { return out_; }
  // Average -log2 P(chosen byte) under the blended model: how surprising the
  // output is to the model itself.
  double bits_per_byte() const { return out_.empty() ? 0.0 : bits_ / (double)out_.size(); }
  double total_bits() const { return bits_; }
  // Recent bits/byte the source models spent on their training text:
  // how surprising real text is to them once trained.
  double training_bits_per_byte() const;

  // Everything needed to roll generation back (best-of-N).
  struct Snapshot {
    std::vector<Stream> streams;
    std::vector<uint64_t> hist_end;
    std::vector<std::unique_ptr<Constraint>> constraints;
    std::mt19937_64 rng;
    size_t out_size;
    double bits;
  };
  Snapshot snapshot() const;
  void restore(const Snapshot& s);

 private:
  void advance(uint8_t b);
  void plan_units();
  double uniform();
  std::vector<Source> src_;
  GenOptions opt_;
  std::vector<std::unique_ptr<Constraint>> constraints_;
  std::mt19937_64 rng_;
  std::vector<uint8_t> out_;
  double bits_ = 0;
};

}  // namespace cmix
