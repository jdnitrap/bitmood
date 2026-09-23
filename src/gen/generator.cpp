#include "gen/generator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "core/math.h"
#include "model/session.h"

namespace cmix {

namespace {
// Room kept free in each model's history so a generation run never triggers
// a drop of old history (which would make snapshots impossible to restore).
constexpr size_t kHistoryRoom = size_t(1) << 18;
}  // namespace

Generator::Generator(std::vector<Source> sources, GenOptions opt)
    : src_(std::move(sources)), opt_(opt), rng_(opt.seed) {
  if (src_.empty()) throw std::runtime_error("generator needs at least one model");
  if (!(opt_.temp > 0)) throw std::runtime_error("--temp must be > 0");
  for (Source& s : src_) s.model->history().make_room(kHistoryRoom);
}

double Generator::uniform() { return (double)(rng_() >> 11) * 0x1.0p-53; }

void Generator::advance(uint8_t b) {
  for (Source& s : src_) s.model->advance_byte(s.stream, b);
  for (auto& c : constraints_) c->accept(b);
}

void Generator::feed(uint8_t b) { advance(b); }

bool Generator::must_continue() const {
  for (const auto& c : constraints_)
    if (c->must_continue()) return true;
  return false;
}

namespace {

// node[(1 << depth) | partial] = P(next bit = 1) after `partial` (depth bits)
// -> probability of each of the 256 bytes.
void bytes_from_nodes(const double* node, ByteProbs& p) {
  for (int c = 0; c < 256; ++c) {
    double q = 1.0;
    for (int depth = 0; depth < 8; ++depth) {
      const double p1 = node[(1 << depth) | (c >> (8 - depth))];
      q *= ((c >> (7 - depth)) & 1) ? p1 : 1.0 - p1;
    }
    p[c] = q;
  }
}

// Calls f(depth, partial, BitPos) for all 255 nodes of the bit tree.
template <class F>
void for_each_node(F f) {
  for (int depth = 0; depth < 8; ++depth)
    for (int partial = 0; partial < (1 << depth); ++partial) {
      BitPos bp;
      bp.index = depth;
      bp.partial = partial;
      f((1 << depth) | partial, bp);
    }
}

}  // namespace

double Generator::satisfaction() const {
  double s = 0;
  for (const auto& c : constraints_) s += c->satisfaction();
  return s;
}

void Generator::distribution(ByteProbs& p) const {
  double wsum = 0;
  for (const Source& s : src_) wsum += s.weight;
  if (!(wsum > 0)) throw std::runtime_error("blend weights must sum to more than 0");
  double node[256];
  Model::Votes v;
  if (opt_.blend == BlendMode::Product && src_.size() > 1) {
    for_each_node([&](int id, BitPos bp) {
      double z = 0;
      for (const Source& s : src_) z += s.weight / wsum * stretch(s.model->predict(s.stream, bp, v));
      node[id] = squash((float)z);
    });
    bytes_from_nodes(node, p);
    return;
  }
  p.fill(0.0);
  ByteProbs one;
  for (const Source& s : src_) {
    if (s.weight <= 0) continue;
    for_each_node([&](int id, BitPos bp) { node[id] = s.model->predict(s.stream, bp, v) / 65536.0; });
    bytes_from_nodes(node, one);
    for (int c = 0; c < 256; ++c) p[c] += s.weight / wsum * one[c];
  }
}

ByteMask Generator::allowed() const {
  auto any = [](const ByteMask& m) { return std::any_of(m.begin(), m.end(), [](bool b) { return b; }); };
  ByteMask mask;
  mask.fill(true);
  for (const auto& c : constraints_)
    if (!c->soft()) c->restrict(mask);
  if (!any(mask)) throw std::runtime_error("the constraints leave no byte that may come next");
  std::vector<const Constraint*> soft;
  for (const auto& c : constraints_)
    if (c->soft()) soft.push_back(c.get());
  std::stable_sort(soft.begin(), soft.end(), [](const Constraint* a, const Constraint* b) {
    return a->priority() > b->priority();
  });
  for (const Constraint* c : soft) {
    ByteMask trial = mask;
    c->restrict(trial);
    if (any(trial)) mask = trial;
  }
  return mask;
}

void Generator::plan_units() {
  for (Source& s : src_) {
    Model& m = *s.model;
    if (!m.graph_can_plan(s.stream)) continue;
    const auto cands = m.graph_candidates(s.stream);
    double total = 0;
    std::vector<double> w;
    for (const auto& c : cands) {
      w.push_back(std::pow(c.weight, 1.0 / opt_.temp));
      total += w.back();
    }
    Token pick = kNoToken;
    if (total > 0) {
      double r = uniform() * total;
      pick = cands.back().token;
      for (size_t i = 0; i < cands.size(); ++i) {
        if (r < w[i]) {
          pick = cands[i].token;
          break;
        }
        r -= w[i];
      }
    }
    m.replan(s.stream, pick);
  }
}

uint8_t Generator::next() {
  if (opt_.plan) plan_units();
  ByteProbs model_p;
  distribution(model_p);
  // Steer toward planned words: boost the byte that continues each source's plan.
  ByteProbs steer = model_p;
  if (opt_.plan && opt_.plan_strength > 0)
    for (const Source& s : src_) {
      const int c = s.model->plan_next_byte(s.stream);
      if (c >= 0) steer[c] *= 1.0 + opt_.plan_strength;
    }
  const ByteMask ok = allowed();

  // Candidates: allowed bytes with their temperature-adjusted weight.
  std::vector<std::pair<double, int>> cand;
  for (int c = 0; c < 256; ++c)
    if (ok[c]) cand.push_back({std::pow(std::max(steer[c], 1e-300), 1.0 / opt_.temp), c});
  std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  if (opt_.top_k > 0 && (int)cand.size() > opt_.top_k) cand.resize((size_t)opt_.top_k);
  double total = 0;
  for (const auto& x : cand) total += x.first;
  if (opt_.top_p < 1.0) {
    double run = 0;
    size_t keep = 0;
    while (keep < cand.size() && run < opt_.top_p * total) run += cand[keep++].first;
    cand.resize(std::max<size_t>(keep, 1));
    total = run;
  }

  double r = uniform() * total;
  int chosen = cand.back().second;
  for (const auto& x : cand) {
    if (r < x.first) {
      chosen = x.second;
      break;
    }
    r -= x.first;
  }
  bits_ += -std::log2(std::max(model_p[chosen], 1e-300));
  out_.push_back((uint8_t)chosen);
  advance((uint8_t)chosen);
  return (uint8_t)chosen;
}

void Generator::learn_byte(uint8_t b) {
  for (Source& s : src_) {
    Session sess(*s.model, s.stream);
    sess.learn_byte(b);
    s.stream = sess.stream();
  }
  for (auto& c : constraints_) c->accept(b);
}

void Generator::ensure_room() {
  for (Source& s : src_) s.model->history().make_room(kHistoryRoom);
}

void Generator::emit(uint8_t b, double bits) {
  bits_ += bits;
  out_.push_back(b);
  advance(b);
}

double Generator::training_bits_per_byte() const {
  double w = 0, sum = 0;
  for (const Source& s : src_) {
    const Model& m = *s.model;
    if (m.bytes_learned == 0) continue;
    sum += s.weight * m.recent_bpb;
    w += s.weight;
  }
  return w > 0 ? sum / w : 0.0;
}

Generator::Snapshot Generator::snapshot() const {
  Snapshot s;
  for (const Source& src : src_) {
    s.streams.push_back(src.stream);
    s.hist_end.push_back(src.model->history().end());
  }
  for (const auto& c : constraints_) s.constraints.push_back(c->clone());
  s.rng = rng_;
  s.out_size = out_.size();
  s.bits = bits_;
  return s;
}

void Generator::restore(const Snapshot& s) {
  for (size_t i = 0; i < src_.size(); ++i) {
    if (s.hist_end[i] < src_[i].model->history().base())
      throw std::runtime_error("generation ran past the history room; cannot roll back");
    src_[i].stream = s.streams[i];
    src_[i].model->history().truncate_to(s.hist_end[i]);
  }
  constraints_.clear();
  for (const auto& c : s.constraints) constraints_.push_back(c->clone());
  rng_ = s.rng;
  out_.resize(s.out_size);
  bits_ = s.bits;
}

}  // namespace cmix
