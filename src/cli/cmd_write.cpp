// write: type with the model. It suggests the next few bytes in grey;
// Tab accepts them, the right arrow accepts one word, and every finished
// word is learned. The memory is saved on Ctrl-S and on quit.
#include <cstdio>
#include <stdexcept>

#include "cli/commands.h"
#include "cli/common.h"
#include "gen/writer.h"
#include "io/files.h"
#include "io/state.h"
#include "term/terminal.h"

namespace cmix {

namespace {

// Number of terminal columns a UTF-8 string takes (1 per code point).
size_t columns(const std::string& s) {
  size_t n = 0;
  for (unsigned char c : s)
    if ((c & 0xC0) != 0x80) ++n;
  return n;
}

// The last `cols` columns of s, starting at a code point boundary.
std::string tail_columns(const std::string& s, size_t cols) {
  size_t n = 0, i = s.size();
  while (i > 0) {
    --i;
    if (((unsigned char)s[i] & 0xC0) != 0x80 && ++n == cols) break;
  }
  return s.substr(i);
}

// The first `cols` columns of s.
std::string head_columns(const std::string& s, size_t cols) {
  size_t n = 0, i = 0;
  for (; i < s.size(); ++i) {
    if (((unsigned char)s[i] & 0xC0) != 0x80) {
      if (n == cols) break;
      ++n;
    }
  }
  return s.substr(0, i);
}

void redraw(RawTerminal& term, const WritingSession& w) {
  const std::string& t = w.text();
  const size_t nl = t.rfind('\n');
  const std::string line = nl == std::string::npos ? t : t.substr(nl + 1);
  const size_t width = (size_t)std::max(20, term.width() - 1);
  const std::string sugg = head_columns(w.suggestion(), width / 3);
  const std::string shown = tail_columns(line, width - columns(sugg));
  std::string out = "\r\x1b[K" + shown;
  if (!sugg.empty()) out += "\x1b[90m" + sugg + "\x1b[0m\x1b[" + std::to_string(columns(sugg)) + "D";
  term.write(out);
}

}  // namespace

int cmd_write(int argc, char** argv, int start) {
  Args a(argc, argv, start, {"state", "type", "out", "suggest", "charset"}, {"no-save"});
  if (!a.has("state") || !a.pos().empty())
    throw std::runtime_error(
        "usage: write --state <memory.bin> [--out text.txt] [--suggest N] [--charset seen|utf8|ascii|any] [--no-save]");
  const std::string path = a.str("state");
  const long long suggest = a.integer("suggest", 24);
  if (suggest < 0) throw std::runtime_error("--suggest must be >= 0");

  Stream st;
  bool created = false;
  auto model = open_or_create(path, a, st, created);
  GenOptions opt;
  opt.top_k = 1;  // greedy suggestions
  Generator gen({{model.get(), st, 1.0}}, opt);
  GenSetup setup;  // for the charset constraint: which bytes the memory has seen
  setup.sources.push_back({model.get(), st, 1.0});
  add_standard_constraints(gen, a, setup);

  WritingSession w(gen, (size_t)suggest);
  {
    RawTerminal term;
    term.write("\x1b[2m" + std::string(created ? "new memory " : "memory ") + path +
               " | Tab: accept  \xE2\x86\x92: accept word  Ctrl-S: save  Ctrl-D/Esc: save and quit\x1b[0m\r\n");
    redraw(term, w);
    for (;;) {
      KeyEvent k = term.read_key();
      if (k.key == Key::Quit) break;
      switch (k.key) {
        case Key::Char: w.type(k.byte); break;
        case Key::Enter:
          w.type('\n');
          term.write("\r\n");
          break;
        case Key::Tab: w.accept_suggestion(false); break;
        case Key::Right: w.accept_suggestion(true); break;
        case Key::Backspace:
          if (!w.backspace()) term.write("\a");  // finished words are already learned
          break;
        case Key::Save:
          w.commit();
          if (!a.has("no-save")) save_state(path, gen.model(0), gen.stream(0));
          break;
        default: break;
      }
      redraw(term, w);
    }
    w.commit();
    term.write("\r\x1b[K" + tail_columns(w.text().substr(w.text().rfind('\n') == std::string::npos
                                                                ? 0
                                                                : w.text().rfind('\n') + 1),
                                         (size_t)std::max(20, term.width() - 1)) +
               "\r\n");
  }
  if (!a.has("no-save")) save_state(path, gen.model(0), gen.stream(0));
  if (a.has("out")) write_file_atomic(a.str("out"), std::vector<uint8_t>(w.text().begin(), w.text().end()));
  std::fprintf(stderr, "learned %llu bytes%s%s\n", (unsigned long long)w.learned(),
               a.has("no-save") ? "" : ("; saved " + path).c_str(),
               a.has("out") ? ("; text in " + a.str("out")).c_str() : "");
  return 0;
}

}  // namespace cmix
