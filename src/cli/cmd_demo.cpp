// demo: print every bit's specialist votes, the mixed p, and which mixer
// weight set (grid view) is active.
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
  for (int i = 0; i < n; ++i) std::printf("%5s", m.input_name(i).c_str());
  std::printf("  set  ch\n");
  for (unsigned char ch : text) {
    for (int k = 0; k < 8; ++k) {
      int bit = (ch >> (7 - k)) & 1;
      int p = s.predict();
      const auto& v = s.votes();
      std::printf("%d    %4d ", bit, p);
      for (int i = 0; i < n; ++i) std::printf("%5d", v.p[i]);
      std::printf("  %3d  %c\n", v.set, (k == 7 && ch >= 32 && ch < 127) ? ch : ' ');
      s.learn_bit(bit);
    }
  }
  std::printf("final mixer weights (set %d):", m.best_view());
  for (int i = 0; i < n; ++i) std::printf(" %s=%d", m.input_name(i).c_str(), m.mixer().weight(i, m.best_view()));
  std::printf("\n");
  return 0;
}

}  // namespace cmix
