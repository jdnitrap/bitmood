// compare: side-by-side report on two files of the same type.
//
// For each file, a fresh model learns it once (as compression would) and we
// record: bits per byte, how the same file does without grid views, how
// well each specialist would do alone, and which grid view won where
// (segments). Then cross-file: how much learning one file helps the other.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "cli/args.h"
#include "cli/commands.h"
#include "cli/common.h"
#include "core/math.h"
#include "io/files.h"
#include "io/formats.h"
#include "model/session.h"

namespace cmix {

namespace {

struct Segment {
  uint64_t start, end;
  std::string label;
};

struct Report {
  std::string name;
  uint64_t bytes = 0;
  double bits = 0;
  double bits_no_grid = 0;
  std::vector<double> solo_bits;       // per input: cost if it predicted alone
  std::vector<std::string> input_labels;
  std::vector<int> final_widths;  // auto widths at the end of the file
  std::vector<Segment> segments;
  std::vector<std::pair<std::string, uint64_t>> view_bytes;  // label -> bytes won
};

// Builds segments from a per-byte label, ignoring runs shorter than kMinRun.
class SegmentBuilder {
 public:
  static constexpr uint64_t kMinRun = 64;
  void add(uint64_t pos, const std::string& label) {
    if (segs_.empty()) {
      segs_.push_back({pos, pos + 1, label});
      return;
    }
    Segment& cur = segs_.back();
    cur.end = pos + 1;
    if (label == cur.label) {
      cand_.clear();
      return;
    }
    if (label != cand_) {
      cand_ = label;
      cand_start_ = pos;
      return;
    }
    if (pos + 1 - cand_start_ >= kMinRun) {
      cur.end = cand_start_;
      segs_.push_back({cand_start_, pos + 1, cand_});
      cand_.clear();
    }
  }
  std::vector<Segment> take() { return std::move(segs_); }

 private:
  std::vector<Segment> segs_;
  std::string cand_;
  uint64_t cand_start_ = 0;
};

std::string best_view_label(const Model& m, const Stream& s) {
  int v = m.best_view();
  return v == 0 ? "no grid" : view_label(m.grid().view(v - 1), s);
}

// Learns `data` with session s as one record; returns bits spent.
// Optionally fills the report.
double learn_all(Session& s, const std::vector<uint8_t>& data, Report* r) {
  Model& m = s.model();
  s.begin_record();
  const int n = m.num_inputs();
  SegmentBuilder segs;
  std::vector<std::pair<std::string, uint64_t>> wins;
  if (r) r->solo_bits.assign(n, 0.0);
  double bits = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    if (r) {
      std::string label = best_view_label(m, s.stream());
      segs.add(i, label);
      auto it = std::find_if(wins.begin(), wins.end(), [&](const auto& w) { return w.first == label; });
      if (it == wins.end()) wins.push_back({label, 1});
      else ++it->second;
    }
    for (int k = 7; k >= 0; --k) {
      const int bit = (data[i] >> k) & 1;
      s.predict();
      if (r)
        for (int j = 0; j < n; ++j) r->solo_bits[j] += bit_cost(s.votes().p[j], bit);
      bits += s.learn_bit(bit);
    }
  }
  if (r) {
    for (int v = 0; v < m.grid().num_views(); ++v)
      if (m.grid().view(v).kind == ViewKind::Auto) r->final_widths.push_back(s.stream().widths.width[m.grid().view(v).param]);
    for (int j = 0; j < n; ++j) r->input_labels.push_back(m.input_label(j));
    r->segments = segs.take();
    std::sort(wins.begin(), wins.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    r->view_bytes = wins;
  }
  return bits;
}

std::string base_name(const std::string& p) {
  size_t k = p.find_last_of('/');
  return k == std::string::npos ? p : p.substr(k + 1);
}

std::string bar(double bpb) {
  // 8 blocks; fuller = cheaper (better). 0 bits/byte = full, 8+ = empty.
  int full = (int)(8.0 * (1.0 - std::min(bpb, 8.0) / 8.0) + 0.5);
  std::string s;
  for (int i = 0; i < 8; ++i) s += i < full ? "█" : "░";
  return s;
}

}  // namespace

int cmd_compare(int argc, char** argv, int start) {
  Args a(argc, argv, start, {"type", "width", "channels", "lstm", "graph-confirm", "snn-leak"}, {"graph", "snn"});
  if (a.pos().size() != 2)
    throw std::runtime_error(
        "usage: compare [--type text|image|audio|raw] [--width W --channels C] [--lstm N] [--graph] <fileA> <fileB>");
  const Config base = config_from_args(a);

  std::vector<uint8_t> data[2];
  Config cfg[2] = {base, base};
  Report rep[2];
  for (int f = 0; f < 2; ++f) {
    TypedData d = read_typed(a.pos()[f], base.type, true);
    apply_shape(cfg[f], d, true, a.pos()[f]);
    data[f] = std::move(d.payload);
    rep[f].name = base_name(a.pos()[f]);
    rep[f].bytes = data[f].size();
    {
      Model m(cfg[f]);
      Session s(m);
      rep[f].bits = learn_all(s, data[f], &rep[f]);
    }
    {
      Config no_grid = cfg[f];
      no_grid.grid = false;
      Model m(no_grid);
      Session s(m);
      rep[f].bits_no_grid = learn_all(s, data[f], nullptr);
    }
  }
  // cross[f]: bits for file f after first learning the other file (same shape only).
  const bool same_shape = cfg[0].width == cfg[1].width && cfg[0].channels == cfg[1].channels;
  double cross[2] = {0, 0};
  for (int f = 0; f < 2 && same_shape; ++f) {
    Model m(cfg[f]);
    Session s(m);
    learn_all(s, data[1 - f], nullptr);
    cross[f] = learn_all(s, data[f], nullptr);
  }

  const int W = 22;
  std::printf("%-*s %-24s %-24s\n", W, "", rep[0].name.c_str(), rep[1].name.c_str());
  std::printf("%-*s %-24llu %-24llu\n", W, "size (bytes)", (unsigned long long)rep[0].bytes,
              (unsigned long long)rep[1].bytes);
  std::printf("%-*s %-24.3f %-24.3f  <- lower = more predictable\n", W, "bits per byte",
              per_byte(rep[0].bits, rep[0].bytes), per_byte(rep[1].bits, rep[1].bytes));
  std::printf("%-*s %-24.3f %-24.3f\n", W, "  without grid views", per_byte(rep[0].bits_no_grid, rep[0].bytes),
              per_byte(rep[1].bits_no_grid, rep[1].bytes));
  auto gain = [](double with, double without) { return without > 0 ? 100.0 * (without - with) / without : 0.0; };
  char g0[32], g1[32];
  std::snprintf(g0, sizeof g0, "%.1f%%", gain(rep[0].bits, rep[0].bits_no_grid));
  std::snprintf(g1, sizeof g1, "%.1f%%", gain(rep[1].bits, rep[1].bits_no_grid));
  std::printf("%-*s %-24s %-24s\n", W, "  grid saves", g0, g1);
  std::printf("%-*s %-24s %-24s\n", W, "view that won most",
              rep[0].view_bytes.empty() ? "-" : rep[0].view_bytes[0].first.c_str(),
              rep[1].view_bytes.empty() ? "-" : rep[1].view_bytes[0].first.c_str());

  std::printf("\nspecialists alone (bits/byte, fuller bar = better):\n");
  for (size_t j = 0; j < rep[0].solo_bits.size(); ++j) {
    char c0[64], c1[64];
    std::snprintf(c0, sizeof c0, "%s %5.2f", bar(per_byte(rep[0].solo_bits[j], rep[0].bytes)).c_str(),
                  per_byte(rep[0].solo_bits[j], rep[0].bytes));
    std::snprintf(c1, sizeof c1, "%s %5.2f", bar(per_byte(rep[1].solo_bits[j], rep[1].bytes)).c_str(),
                  per_byte(rep[1].solo_bits[j], rep[1].bytes));
    std::printf("  %-28s %s    %s\n", rep[0].input_labels[j].c_str(), c0, c1);
  }
  auto widths = [](const Report& r) {
    std::string s;
    for (int w : r.final_widths) s += (s.empty() ? "" : ", ") + (w ? std::to_string(w) : std::string("none"));
    return s.empty() ? std::string("-") : s;
  };
  std::printf("  %-28s %-24s %-24s\n", "auto widths at end of file", widths(rep[0]).c_str(), widths(rep[1]).c_str());

  for (int f = 0; f < 2; ++f) {
    std::printf("\nsegments in %s (best view, runs of %llu+ bytes):\n", rep[f].name.c_str(),
                (unsigned long long)SegmentBuilder::kMinRun);
    const auto& segs = rep[f].segments;
    const size_t show = std::min<size_t>(segs.size(), 16);
    for (size_t i = 0; i < show; ++i)
      std::printf("  0x%06llx-0x%06llx  %7llu bytes  %s\n", (unsigned long long)segs[i].start,
                  (unsigned long long)segs[i].end, (unsigned long long)(segs[i].end - segs[i].start),
                  segs[i].label.c_str());
    if (segs.size() > show) std::printf("  ... %zu more\n", segs.size() - show);
    std::printf("  totals:");
    for (const auto& v : rep[f].view_bytes)
      std::printf("  %s %.0f%%", v.first.c_str(), 100.0 * v.second / std::max<uint64_t>(1, rep[f].bytes));
    std::printf("\n");
  }

  std::printf("\ncross-file (does learning one file help the other?):\n");
  if (!same_shape) std::printf("  skipped: the files have different shapes\n");
  for (int f = 0; f < 2 && same_shape; ++f) {
    double cold = per_byte(rep[f].bits, rep[f].bytes), warm = per_byte(cross[f], rep[f].bytes);
    std::printf("  %-24s cold %.3f  after %-24s %.3f bits/byte  (%+.1f%%)\n", rep[f].name.c_str(), cold,
                rep[1 - f].name.c_str(), warm, cold > 0 ? 100.0 * (warm - cold) / cold : 0.0);
  }
  return 0;
}

}  // namespace cmix
