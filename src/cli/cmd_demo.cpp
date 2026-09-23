// demo: print every bit's specialist votes, the mixed p, and mixer weights.
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
  std::cout << "bit  p(1)   A    B    C   D0   D1   D2   wA wB wC ...  ch\n";
  for (unsigned char ch : text) {
    for (int k = 0; k < 8; ++k) {
      int bit = (ch >> (7 - k)) & 1;
      int p = s.predict();
      const auto& v = s.votes();
      const Mixer& mx = m.mixer();
      std::printf("%d    %4d  %4d %4d %4d %4d %4d %4d  %3d %3d %3d     %c\n", bit, p, v.p[0], v.p[1], v.p[2],
                  v.p[3], v.p[4], v.p[5], mx.weight(0), mx.weight(1), mx.weight(2),
                  (k == 7 && ch >= 32 && ch < 127) ? ch : ' ');
      s.learn_bit(bit);
    }
  }
  std::cout << "final mixer weights:";
  for (int i = 0; i < m.mixer().size(); ++i) std::cout << " " << m.mixer().weight(i);
  std::cout << "\n";
  return 0;
}

}  // namespace cmix
