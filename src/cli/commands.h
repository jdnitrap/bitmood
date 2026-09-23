// One function per CLI command. Each takes the raw argv and the index where
// its own arguments start, and returns the process exit code.
#pragma once

namespace cmix {

int cmd_compress(int argc, char** argv, int start);
int cmd_decompress(int argc, char** argv, int start);
int cmd_demo(int argc, char** argv, int start);
int cmd_train(int argc, char** argv, int start);
int cmd_generate(int argc, char** argv, int start);
int cmd_info(int argc, char** argv, int start);
int cmd_compare(int argc, char** argv, int start);

}  // namespace cmix
