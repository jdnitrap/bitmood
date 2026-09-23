#include "cli/common.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "gen/shape.h"
#include "io/files.h"
#include "io/state.h"
#include "model/session.h"

namespace cmix {

Config config_from_args(const Args& a) {
  Config cfg;
  cfg.type = parse_type(a.str("type", "text"));
  if (a.has("table-bits")) {
    long long b = a.integer("table-bits", 22);
    if (b < 16 || b > 28) throw std::runtime_error("--table-bits must be 16..28");
    cfg.table_bits = (int)b;
  }
  if (a.has("width")) cfg.width = (int)a.integer("width", 0);
  if (a.has("channels")) cfg.channels = (int)a.integer("channels", 0);
  if (a.has("sample-rate")) cfg.sample_rate = (int)a.integer("sample-rate", 0);
  if (a.has("graph")) cfg.graph = true;
  if (a.has("graph-confirm")) cfg.graph_confirm = (int)a.integer("graph-confirm", 2);
  if (a.has("lstm")) {
    long long n = a.integer("lstm", 0);
    if (n < 0 || n > LstmState::kMaxCells) throw std::runtime_error("--lstm must be 0..64 cells");
    cfg.lstm_cells = (int)n;
  }
  return cfg;
}

std::unique_ptr<Model> open_or_create(const std::string& path, const Args& a, Stream& s, bool& created,
                                      const TypedData* first) {
  created = !file_exists(path);
  if (created) {
    Config cfg = config_from_args(a);
    if (first) apply_shape(cfg, *first, true, "the first file");
    else if (cfg.type == DataType::Image || cfg.type == DataType::Audio) apply_shape(cfg, TypedData(), true, "");
    s = Stream();
    return std::make_unique<Model>(cfg);
  }
  auto m = load_state(path, s);
  if ((a.has("graph") || a.has("graph-confirm")) && !m->config().graph)
    throw std::runtime_error(path + " already exists without a graph; --graph only applies to new memories");
  if (a.has("lstm") && a.integer("lstm", 0) != m->config().lstm_cells)
    throw std::runtime_error(path + " already exists with --lstm " + std::to_string(m->config().lstm_cells));
  if (a.has("table-bits") && a.integer("table-bits", 0) != m->config().table_bits)
    throw std::runtime_error(path + " already exists with --table-bits " + std::to_string(m->config().table_bits));
  if (a.has("type") && parse_type(a.str("type")) != m->config().type)
    throw std::runtime_error(path + " is a " + type_name(m->config().type) + " memory, not " + a.str("type"));
  return m;
}

namespace {

std::vector<double> parse_blend(const std::string& s, size_t n) {
  std::vector<double> w;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) w.push_back(std::stod(item));
  if (w.size() != n)
    throw std::runtime_error("--blend needs one weight per --state (" + std::to_string(n) + ")");
  for (double x : w)
    if (x < 0) throw std::runtime_error("--blend weights must be >= 0");
  return w;
}

}  // namespace

GenSetup load_sources(const Args& a, const std::string& prompt, bool need_training_text) {
  if (a.has("graph-plan") && a.all("state").empty()) throw std::runtime_error("--graph-plan needs a memory made with --graph");
  GenSetup g;
  const std::vector<std::string> paths = a.all("state");
  std::vector<double> weights(paths.size(), 1.0);
  if (a.has("blend")) weights = parse_blend(a.str("blend"), paths.size());

  if (paths.empty()) {
    // No memory: the prompt is all the model has, so it learns from it.
    auto m = std::make_unique<Model>(Config());
    Session s(*m);
    for (unsigned char ch : prompt) s.learn_byte(ch);
    g.sources.push_back({m.get(), s.stream(), 1.0});
    g.models.push_back(std::move(m));
    if (need_training_text) g.training = std::make_shared<SuffixArray>(std::vector<uint8_t>());
    return g;
  }

  std::vector<uint8_t> text;
  for (size_t i = 0; i < paths.size(); ++i) {
    Stream st;
    auto m = load_state(paths[i], st);
    if (i > 0 && m->config().type != g.models[0]->config().type)
      throw std::runtime_error("cannot blend a " + std::string(type_name(m->config().type)) + " memory with a " +
                               type_name(g.models[0]->config().type) + " memory");
    if (need_training_text) {
      text.insert(text.end(), m->history().data().begin(), m->history().data().end());
      text.push_back(0);  // separator so no copy spans two memories
    }
    // With a memory the prompt only sets the context. Images and sounds
    // start a new record: generation begins at pixel / sample 0.
    Session s(*m, st);
    if (m->config().type == DataType::Image || m->config().type == DataType::Audio) s.begin_record();
    for (unsigned char ch : prompt) s.feed_byte(ch);
    if (a.has("graph-plan") && !m->graph())
      throw std::runtime_error(paths[i] + " has no graph (train a new memory with --graph)");
    g.sources.push_back({m.get(), s.stream(), weights[i]});
    g.models.push_back(std::move(m));
  }
  if (need_training_text) g.training = std::make_shared<SuffixArray>(std::move(text));
  return g;
}

GenOptions gen_options(const Args& a) {
  GenOptions o;
  const std::string mode = a.str("blend-mode", "mix");
  if (mode == "mix") o.blend = BlendMode::Mix;
  else if (mode == "product") o.blend = BlendMode::Product;
  else throw std::runtime_error("--blend-mode must be mix or product");
  o.temp = a.num("temp", 1.0);
  o.top_p = a.num("top-p", 1.0);
  o.top_k = (int)a.integer("top-k", 0);
  o.seed = (uint64_t)a.integer("seed", 0xC0FFEE);
  o.plan = a.has("graph-plan") || a.has("plan-strength");
  o.plan_strength = a.num("plan-strength", 4.0);
  if (o.plan_strength < 0) throw std::runtime_error("--plan-strength must be >= 0");
  if (!(o.temp > 0)) throw std::runtime_error("--temp must be > 0");
  if (!(o.top_p > 0 && o.top_p <= 1)) throw std::runtime_error("--top-p must be in (0, 1]");
  if (o.top_k < 0) throw std::runtime_error("--top-k must be >= 0");
  return o;
}

void add_standard_constraints(Generator& g, const Args& a, const GenSetup& setup) {
  ByteMask seen{};
  for (const Source& s : setup.sources)
    for (uint8_t b : s.model->history().data()) seen[b] = true;
  const bool text = setup.sources[0].model->config().type == DataType::Text;
  // A memory that has seen nothing yet can't use "seen"; fall back to utf8.
  const bool any_seen = std::find(seen.begin(), seen.end(), true) != seen.end();
  const std::string def = text ? (any_seen ? "seen" : "utf8") : "any";
  g.add_constraint(std::make_unique<CharsetFilter>(CharsetFilter::parse(a.str("charset", def)), seen));
  const long long novelty = a.integer("novelty", 0);
  if (novelty < 0) throw std::runtime_error("--novelty must be >= 0");
  if (novelty > 0) g.add_constraint(std::make_unique<NoveltyFilter>(setup.training, (int)novelty));
}

void add_shape_constraints(Generator& g, const Args& a) {
  if (a.has("line-start") && a.has("acrostic")) throw std::runtime_error("use --line-start or --acrostic, not both");
  if (a.has("line-start")) g.add_constraint(std::make_unique<LineStartFilter>(a.str("line-start"), false));
  if (a.has("acrostic")) g.add_constraint(std::make_unique<LineStartFilter>(a.str("acrostic"), true));
  if (a.has("max-line")) {
    long long n = a.integer("max-line", 0);
    if (n < 1) throw std::runtime_error("--max-line must be >= 1");
    g.add_constraint(std::make_unique<MaxLineFilter>((int)n));
  }
  if (a.has("words")) g.add_constraint(std::make_unique<WordListFilter>(WordListFilter::load(a.str("words"))));
  if (a.has("rhyme")) g.add_constraint(std::make_unique<RhymeFilter>());
}

const std::set<std::string> kGenValued = {"state",   "blend",      "blend-mode", "temp",     "top-p",    "top-k",
                                          "seed",    "charset",    "novelty",  "line-start", "acrostic",
                                          "max-line", "words",     "best-of",  "out",      "height",
                                          "seconds",  "plan-strength"};
const std::set<std::string> kGenFlags = {"stats", "rhyme", "graph-plan"};

}  // namespace cmix
