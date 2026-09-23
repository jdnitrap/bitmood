// cmix-bit: context-mixing bit predictor. Command dispatch only.
#include <cstring>
#include <exception>
#include <iostream>

#include "cli/commands.h"

namespace {

void usage() {
  std::cerr << "cmix-bit -- context-mixing bit predictor\n"
               "\n"
               "  cmix-bit compress   [--type text|image|audio|raw] [--width W --channels C] [--lstm N]\n                      <in> <out.cmxb>\n"
               "  cmix-bit decompress <in.cmxb> <out>\n"
               "  cmix-bit demo       [text]\n"
               "  cmix-bit train      --state <memory.bin> [--type text|image|audio|raw] [--table-bits 16..28] [--lstm N]\n"
               "                      [--width W --channels 1|3] <file>...\n"
               "  cmix-bit generate   <nbytes> [prompt] [--state memory.bin]... [--blend w1,w2,..] [--blend-mode mix|product]\n"
               "                      [--temp T] [--top-p P] [--top-k K] [--seed N]\n"
               "                      [--charset seen|utf8|ascii|any] [--novelty N]\n"
               "                      [--line-start CHARS | --acrostic WORD] [--max-line N]\n"
               "                      [--words FILE] [--rhyme] [--best-of N] [--stats]\n"
               "  cmix-bit info       <memory.bin>\n"
               "  cmix-bit compare    [--type text|image|audio|raw] <fileA> <fileB>\n"
               "  cmix-bit write      --state <memory.bin> [--out text.txt] [--suggest N] [--no-save]\n";
}

struct Command {
  const char* name;
  int (*fn)(int, char**, int);
};

const Command kCommands[] = {
    {"compress", cmix::cmd_compress}, {"decompress", cmix::cmd_decompress}, {"demo", cmix::cmd_demo},
    {"train", cmix::cmd_train},       {"generate", cmix::cmd_generate},     {"info", cmix::cmd_info},
    {"compare", cmix::cmd_compare},   {"write", cmix::cmd_write},
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
