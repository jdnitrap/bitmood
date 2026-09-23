// cmix-bit: context-mixing bit predictor. Command dispatch only.
#include <cstring>
#include <exception>
#include <iostream>

#include "cli/commands.h"

namespace {

void usage() {
  std::cerr << "cmix-bit -- context-mixing bit predictor\n"
               "\n"
               "  cmix-bit compress   <in> <out.cmxb>\n"
               "  cmix-bit decompress <in.cmxb> <out>\n"
               "  cmix-bit demo       [text]\n"
               "  cmix-bit train      --state <memory.bin> [--type text] <file>...\n"
               "  cmix-bit generate   <nbytes> [prompt] [--state memory.bin] [--temp T] [--seed N]\n"
               "  cmix-bit info       <memory.bin>\n";
}

struct Command {
  const char* name;
  int (*fn)(int, char**, int);
};

const Command kCommands[] = {
    {"compress", cmix::cmd_compress}, {"decompress", cmix::cmd_decompress}, {"demo", cmix::cmd_demo},
    {"train", cmix::cmd_train},       {"generate", cmix::cmd_generate},     {"info", cmix::cmd_info},
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 1;
  }
  for (const Command& c : kCommands) {
    if (std::strcmp(argv[1], c.name) != 0) continue;
    try {
      return c.fn(argc, argv, 2);
    } catch (const std::exception& e) {
      std::cerr << "cmix-bit " << c.name << ": " << e.what() << "\n";
      return 1;
    }
  }
  usage();
  return 1;
}
