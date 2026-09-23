#include "cli/args.h"

#include <cstdlib>
#include <stdexcept>

namespace cmix {

Args::Args(int argc, char** argv, int start, const std::set<std::string>& valued,
           const std::set<std::string>& flags) {
  for (int i = start; i < argc; ++i) {
    std::string a = argv[i];
    if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
      std::string name = a.substr(2);
      if (valued.count(name)) {
        if (i + 1 >= argc) throw std::runtime_error("--" + name + " needs a value");
        opts_.emplace(name, argv[++i]);
        order_.emplace_back(name, argv[i]);
      } else if (flags.count(name)) {
        opts_.emplace(name, "");
        order_.emplace_back(name, "");
      } else {
        throw std::runtime_error("unknown option --" + name);
      }
    } else {
      pos_.push_back(a);
    }
  }
}

std::string Args::str(const std::string& name, const std::string& def) const {
  auto it = opts_.find(name);
  return it == opts_.end() ? def : it->second;
}

std::vector<std::string> Args::all(const std::string& name) const {
  std::vector<std::string> v;
  for (const auto& kv : order_)
    if (kv.first == name) v.push_back(kv.second);
  return v;
}

double Args::num(const std::string& name, double def) const {
  if (!has(name)) return def;
  std::string s = str(name);
  char* end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end) throw std::runtime_error("--" + name + ": not a number: " + s);
  return v;
}

long long Args::integer(const std::string& name, long long def) const {
  if (!has(name)) return def;
  std::string s = str(name);
  char* end = nullptr;
  long long v = std::strtoll(s.c_str(), &end, 0);
  if (end == s.c_str() || *end) throw std::runtime_error("--" + name + ": not an integer: " + s);
  return v;
}

}  // namespace cmix
