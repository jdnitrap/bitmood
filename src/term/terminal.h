// Raw terminal input for interactive commands (POSIX termios).
// RawTerminal switches the terminal to byte-at-a-time, no-echo mode and
// restores it when destroyed, including after an exception.
#pragma once

#include <string>

namespace cmix {

enum class Key {
  Char,       // a plain byte (see KeyEvent::byte)
  Enter,
  Tab,
  Backspace,
  Right,      // right arrow
  Save,       // Ctrl-S
  Quit,       // Ctrl-D, Ctrl-C, or Esc on its own
  Other,      // any other escape sequence or control key (ignored)
};

struct KeyEvent {
  Key key = Key::Other;
  unsigned char byte = 0;
};

class RawTerminal {
 public:
  RawTerminal();   // throws if stdin is not a terminal
  ~RawTerminal();  // restores the original settings
  RawTerminal(const RawTerminal&) = delete;
  RawTerminal& operator=(const RawTerminal&) = delete;

  KeyEvent read_key();  // blocks until a key arrives
  int width() const;    // columns, 80 if unknown
  void write(const std::string& s);

 private:
  int read_byte(int timeout_ms);  // -1 on timeout
  struct Saved;
  Saved* saved_;
};

}  // namespace cmix
