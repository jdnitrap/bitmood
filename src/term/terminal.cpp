#include "term/terminal.h"

#include <poll.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <stdexcept>

namespace cmix {

struct RawTerminal::Saved {
  termios orig;
};

RawTerminal::RawTerminal() : saved_(new Saved) {
  if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &saved_->orig) != 0) {
    delete saved_;
    throw std::runtime_error("write needs an interactive terminal");
  }
  termios raw = saved_->orig;
  raw.c_lflag &= ~(tcflag_t)(ICANON | ECHO | ISIG | IEXTEN);
  raw.c_iflag &= ~(tcflag_t)(IXON | ICRNL);
  raw.c_cc[VMIN] = 1;
  raw.c_cc[VTIME] = 0;
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

RawTerminal::~RawTerminal() {
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved_->orig);
  delete saved_;
}

int RawTerminal::read_byte(int timeout_ms) {
  pollfd p{STDIN_FILENO, POLLIN, 0};
  if (poll(&p, 1, timeout_ms) <= 0) return -1;
  unsigned char c;
  return ::read(STDIN_FILENO, &c, 1) == 1 ? c : -1;
}

KeyEvent RawTerminal::read_key() {
  KeyEvent e;
  int c = -1;
  while (c < 0) c = read_byte(-1);
  switch (c) {
    case '\r':
    case '\n': e.key = Key::Enter; return e;
    case '\t': e.key = Key::Tab; return e;
    case 127:
    case 8: e.key = Key::Backspace; return e;
    case 19: e.key = Key::Save; return e;   // Ctrl-S
    case 3:                                 // Ctrl-C
    case 4: e.key = Key::Quit; return e;    // Ctrl-D
    case 27: {
      int n = read_byte(30);
      if (n < 0) {
        e.key = Key::Quit;  // Esc on its own
        return e;
      }
      if (n == '[' || n == 'O') {
        int last = read_byte(30);
        while (last >= 0 && !(last >= 0x40 && last <= 0x7E)) last = read_byte(30);
        e.key = last == 'C' ? Key::Right : Key::Other;
      }
      return e;
    }
    default: break;
  }
  if (c < 32) return e;  // other control keys
  e.key = Key::Char;
  e.byte = (unsigned char)c;
  return e;
}

int RawTerminal::width() const {
  winsize w{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) return w.ws_col;
  return 80;
}

void RawTerminal::write(const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    ssize_t n = ::write(STDOUT_FILENO, s.data() + off, s.size() - off);
    if (n <= 0) break;
    off += (size_t)n;
  }
}

}  // namespace cmix
