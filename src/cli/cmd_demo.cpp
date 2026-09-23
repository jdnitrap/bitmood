// demo: print every bit's specialist votes and the final p, as P(1) in
// thousandths (500 = no opinion), plus the grid view whose weight set is active.
#include <cstdio>
#include <iostream>

#include "cli/args.h"
#include "cli/commands.h"
#include "model/session.h"

namespace cmix {

int cmd_demo(int argc, char** argv, int start) {
  Args a(argc, argv, start, {}, {});
  std::string text = a.pos().empty() ? "Hello, mixer. 12345 repeated repeated." : a.pos()[0];

  Model m{Config()};
  Session s(m);
  const int n = m.num_inputs();
  std::printf("bit  p(1) ");
  for (int i = 0; i < n - 1; ++i) std::printf("%5s", m.input_name(i).c_str());  // skip bias
  std::printf(" view  ch\n");
  for (unsigned char ch : text) {
    for (int k = 0; k < 8; ++k) {
      int bit = (ch >> (7 - k)) & 1;
      int p = s.predict();
      const auto& v = s.votes();
      auto milli = [](int q) { return (q * 1000 + 32768) >> 16; };
      std::printf("%d    %4d ", bit, milli(p));
      for (int i = 0; i < n - 1; ++i) std::printf("%5d", milli(v.p[i]));
      std::printf("  %3d  %c\n", m.best_view(), (k == 7 && ch >= 32 && ch < 127) ? ch : ' ');
      s.learn_bit(bit);
    }
  }
  std::printf("final mixer weights (match-state, previous-byte, grid-view mixers):");
  for (int k = 0; k < Model::kMixers; ++k) std::printf(" %.2f", m.final_mixer().weight(k, 7));
  std::printf("\n");
  return 0;
}

}  // namespace cmix
