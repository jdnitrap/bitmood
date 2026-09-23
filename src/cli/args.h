// Minimal command-line parser: positional arguments plus --name value options
// and --flag switches. Unknown options are errors, so typos don't pass silently.
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

namespace cmix {

class Args {
 public:
  // Parses argv[start..argc). `valued` options take a value; `flags` don't.
  Args(int argc, char** argv, int start, const std::set<std::string>& valued,
       const std::set<std::string>& flags);

  const std::vector<std::string>& pos() const { return pos_; }
  bool has(const std::string& name) const { return opts_.count(name) > 0; }
  std::string str(const std::string& name, const std::string& def = "") const;
  // All values of a repeatable option, in order (e.g. --state a --state b).
  std::vector<std::string> all(const std::string& name) const;
  double num(const std::string& name, double def) const;
  long long integer(const std::string& name, long long def) const;

 private:
  std::vector<std::string> pos_;
  std::multimap<std::string, std::string> opts_;
  std::vector<std::pair<std::string, std::string>> order_;
};

}  // namespace cmix
