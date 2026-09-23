// graph: look inside a memory's graph.
//   graph MEMORY          summary: size, most frequent words / shapes, strongest links
//   graph MEMORY WORD     (text) what usually follows and precedes WORD
#include <cctype>
#include <cstdio>
#include <stdexcept>

#include "cli/args.h"
#include "cli/commands.h"
#include "graph/shapes.h"
#include "io/state.h"

namespace cmix {

namespace {

std::string name_of(const Model& m, Token t) {
  switch (m.config().type) {
    case DataType::Text: {
      const std::string* w = m.vocab().word(t);
      return w ? *w : "?";
    }
    case DataType::Audio:
      return "[" + describe_audio(t, m.config().sample_rate, m.slice_samples()) + "]";
    case DataType::Image:
      return t == (Token)kImageCodes ? "[row start]" : "[" + describe_image(t) + "]";
    case DataType::Raw: return "?";
  }
  return "?";
}

void print_edges(const Model& m, const std::vector<TokenGraph::Edge>& edges, size_t n) {
  const uint32_t confirm = m.graph()->confirm();
  if (edges.empty()) std::printf("  (none)\n");
  for (size_t i = 0; i < edges.size() && i < n; ++i)
    std::printf("  %6u  %s%s\n", edges[i].count, name_of(m, edges[i].to).c_str(),
                edges[i].count < confirm ? "   (candidate, not confirmed yet)" : "");
}

}  // namespace

int cmd_graph(int argc, char** argv, int start) {
  Args a(argc, argv, start, {"top"}, {});
  if (a.pos().empty() || a.pos().size() > 2) throw std::runtime_error("usage: graph <memory.bin> [word] [--top N]");
  const size_t top = (size_t)a.integer("top", 15);
  Stream st;
  auto m = load_state(a.pos()[0], st);
  const TokenGraph* g = m->graph();
  if (!g) throw std::runtime_error(a.pos()[0] + " has no graph (train a new memory with --graph)");
  const bool text = m->config().type == DataType::Text;

  if (a.pos().size() == 2) {
    if (!text) throw std::runtime_error("looking up a word only works for text memories");
    std::string w;
    for (unsigned char c : a.pos()[1]) w.push_back((char)(c < 0x80 ? std::tolower(c) : c));
    const Token t = word_token(w.data(), (int)w.size());
    if (!m->vocab().word(t)) throw std::runtime_error("the graph has not seen the word '" + w + "'");
    std::printf("'%s' seen %u times\n\nafter '%s':\n", w.c_str(), m->vocab().count(t), w.c_str());
    print_edges(*m, g->after(t), top);
    std::printf("\nbefore '%s':\n", w.c_str());
    print_edges(*m, g->before(t), top);
    return 0;
  }

  const char* what = text ? "word" : (m->config().type == DataType::Image ? "pixel-run" : "sound-shape");
  std::printf("%s graph: %zu nodes, %zu edges, %zu confirmed (an edge needs %u sightings)\n", what, g->nodes(),
              g->edges(false), g->edges(true), g->confirm());
  if (text) {
    std::printf("\nmost frequent words (%zu different):\n", m->vocab().size());
    for (const auto& kv : m->vocab().top(top)) std::printf("  %6u  %s\n", kv.second, name_of(*m, kv.first).c_str());
  }
  std::printf("\nstrongest links:\n");
  for (const auto& l : g->strongest(top))
    std::printf("  %6u  %s -> %s\n", l.count, name_of(*m, l.from).c_str(), name_of(*m, l.to).c_str());
  return 0;
}

}  // namespace cmix
