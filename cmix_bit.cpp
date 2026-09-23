// Context-mixing bit predictor prototype
// Specialists A-D + logistic mixer + FPAQ0-style arithmetic coder.
// CPU-only. Not PAQ-competitive; the wiring is the point.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static inline int clampi(int x, int lo, int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

static int stretch(int p) {
  p = clampi(p, 1, 4094);
  static bool init = false;
  static int tab[4096];
  if (!init) {
    for (int i = 1; i < 4095; ++i) {
      double pr = i / 4095.0;
      double s = std::log(pr / (1.0 - pr));
      tab[i] = clampi((int)std::lround(s * 256.0), -2047, 2047);
    }
    tab[0] = -2047;
    tab[4095] = 2047;
    init = true;
  }
  return tab[p];
}

static int squash(int z) {
  double x = z / 256.0;
  if (x > 20) return 4094;
  if (x < -20) return 1;
  double p = 1.0 / (1.0 + std::exp(-x));
  return clampi((int)std::lround(p * 4095.0), 1, 4094);
}

struct RecentPattern {
  static const int SIZE = 1 << 20;
  std::vector<uint16_t> n0, n1;
  RecentPattern() : n0(SIZE, 1), n1(SIZE, 1) {}
  static uint32_t mix(uint32_t h, uint32_t v) {
    h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h * 0x85ebca6bu;
  }
  int predict(uint32_t ctx) const {
    uint32_t i = ctx & (SIZE - 1);
    int a = n0[i], b = n1[i];
    return clampi((b * 4095) / (a + b), 1, 4094);
  }
  void update(uint32_t ctx, int bit) {
    uint32_t i = ctx & (SIZE - 1);
    if (bit) n1[i] = (uint16_t)std::min(n1[i] + 1, 65535);
    else n0[i] = (uint16_t)std::min(n0[i] + 1, 65535);
    if (n0[i] + n1[i] > 4096) {
      n0[i] = (uint16_t)((n0[i] + 1) >> 1);
      n1[i] = (uint16_t)((n1[i] + 1) >> 1);
    }
  }
};

struct LongMatch {
  std::vector<uint8_t> win;
  size_t pos = 0;
  static const size_t CAP = 1 << 16;
  LongMatch() : win(CAP, 0) {}
  void push_byte(uint8_t b) { win[pos % CAP] = b; ++pos; }
  int predict(const std::vector<uint8_t>& hist, int bit_index, int partial) const {
    if (hist.size() < 3 || pos < 8) return 2048;
    const size_t n = hist.size();
    const size_t max_check = std::min(pos, (size_t)CAP);
    int best = 0;
    int pred = 0;
    for (size_t off = 3; off < max_check && off < 4096; ++off) {
      size_t match = 0;
      while (match < n && match < 64 && match < pos - off) {
        uint8_t older = win[(pos - off - 1 - match) % CAP];
        uint8_t now = hist[n - 1 - match];
        if (older != now) break;
        ++match;
      }
      if ((int)match <= best) continue;
      uint8_t nxt = win[(pos - off) % CAP];
      int have = 8 - bit_index;
      int mask = (0xFF << have) & 0xFF;
      if ((nxt & mask) != ((partial << have) & mask)) continue;
      best = (int)match;
      pred = (nxt >> (7 - bit_index)) & 1;
    }
    if (best < 3) return 2048;
    int conf = std::min(1800, best * 80);
    return pred ? 2048 + conf : 2048 - conf;
  }
};

struct ByteShape {
  static bool is_alpha(int b) {
    return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z');
  }
  static bool is_digit(int b) { return b >= '0' && b <= '9'; }
  static bool is_upper(int b) { return b >= 'A' && b <= 'Z'; }
  static bool is_punct(int b) {
    return (b >= 33 && b <= 47) || (b >= 58 && b <= 64) ||
           (b >= 91 && b <= 96) || (b >= 123 && b <= 126);
  }
  static bool is_space(int b) {
    return b == ' ' || b == '\t' || b == '\n' || b == '\r';
  }
  static bool utf8_lead(int b) { return (b & 0xC0) != 0x80 && b >= 0x80; }
  int predict(int last_byte, int bit_index, int partial) const {
    int p = 2048;
    if (last_byte < 0) return p;
    if (is_alpha(last_byte)) {
      if (bit_index == 0) p -= 400;
      if (bit_index == 1) p += 350;
      if (bit_index == 2 && (partial & 2)) p += 80;
    }
    if (is_digit(last_byte)) {
      if (bit_index == 0) p -= 500;
      if (bit_index == 1) p += 200;
      if (bit_index == 2) p += 400;
      if (bit_index == 3) p += 400;
    }
    if (is_space(last_byte) && bit_index == 0) p -= 300;
    if (is_punct(last_byte) && bit_index == 0) p -= 250;
    if (utf8_lead(last_byte) && bit_index == 0) p += 200;
    (void)partial;
    return clampi(p, 1, 4094);
  }
};

struct OrderN {
  static const int SIZE = 1 << 18;
  std::vector<uint16_t> n0, n1;
  OrderN() : n0(SIZE, 1), n1(SIZE, 1) {}
  int predict(uint32_t h) const {
    uint32_t i = h & (SIZE - 1);
    int a = n0[i], b = n1[i];
    return clampi((b * 4095) / (a + b), 1, 4094);
  }
  void update(uint32_t h, int bit) {
    uint32_t i = h & (SIZE - 1);
    if (bit) n1[i] = (uint16_t)std::min(n1[i] + 1, 65535);
    else n0[i] = (uint16_t)std::min(n0[i] + 1, 65535);
    if (n0[i] + n1[i] > 2048) {
      n0[i] = (uint16_t)((n0[i] + 1) >> 1);
      n1[i] = (uint16_t)((n1[i] + 1) >> 1);
    }
  }
};

struct Mixer {
  static const int N = 6;
  int w[N];
  Mixer() { for (int i = 0; i < N; ++i) w[i] = 128; }
  int mix(const int* p) {
    int z = 0;
    for (int i = 0; i < N; ++i) z += (w[i] * stretch(p[i])) >> 7;
    return squash(z);
  }
  void update(const int* p, int bit, int mixed) {
    int err = (bit ? 4095 : 0) - mixed;
    for (int i = 0; i < N; ++i) {
      int g = (err * stretch(p[i])) >> 16;
      w[i] = clampi(w[i] + g, 1, 1024);
    }
  }
};

struct Predictor {
  RecentPattern A;
  LongMatch B;
  ByteShape C;
  OrderN D0, D1, D2;
  Mixer mix;
  std::vector<uint8_t> hist;
  int last_byte = -1;
  uint32_t h0 = 0, h1 = 0, h2 = 0;
  int bit_index = 0;
  int partial = 0;
  int last_p[6]{};
  int last_mixed = 2048;
  uint32_t ctxA() const {
    uint32_t h = RecentPattern::mix(0xA5A5A5A5u, (uint32_t)last_byte);
    h = RecentPattern::mix(h, (uint32_t)bit_index);
    h = RecentPattern::mix(h, (uint32_t)partial);
    if (!hist.empty()) h = RecentPattern::mix(h, hist.back());
    if (hist.size() >= 2) h = RecentPattern::mix(h, hist[hist.size() - 2]);
    return h;
  }
  uint32_t order_hash(size_t n, uint32_t seed) const {
    uint32_t h = seed;
    for (size_t i = 0; i < n && i < hist.size(); ++i)
      h = RecentPattern::mix(h, hist[hist.size() - 1 - i]);
    return h;
  }
  int predict() {
    last_p[0] = A.predict(ctxA());
    last_p[1] = B.predict(hist, bit_index, partial);
    last_p[2] = C.predict(last_byte, bit_index, partial);
    last_p[3] = D0.predict(h0 ^ (uint32_t)bit_index * 0x9e3779b9u ^ (uint32_t)partial);
    last_p[4] = D1.predict(h1 ^ (uint32_t)(bit_index + 1) * 0x85ebca6bu ^ (uint32_t)partial << 3);
    last_p[5] = D2.predict(h2 ^ ((uint32_t)bit_index << 16) ^ (uint32_t)partial);
    last_mixed = mix.mix(last_p);
    return last_mixed;
  }
  void update(int bit) {
    mix.update(last_p, bit, last_mixed);
    A.update(ctxA(), bit);
    D0.update(h0 ^ (uint32_t)bit_index * 0x9e3779b9u ^ (uint32_t)partial, bit);
    D1.update(h1 ^ (uint32_t)(bit_index + 1) * 0x85ebca6bu ^ (uint32_t)partial << 3, bit);
    D2.update(h2 ^ ((uint32_t)bit_index << 16) ^ (uint32_t)partial, bit);
    partial = (partial << 1) | bit;
    ++bit_index;
    if (bit_index == 8) {
      uint8_t b = (uint8_t)partial;
      B.push_byte(b);
      hist.push_back(b);
      if (hist.size() > 256) hist.erase(hist.begin(), hist.begin() + 64);
      last_byte = b;
      // D0/D1/D2 are order-1/3/4: hash only the last N bytes so contexts recur.
      h0 = order_hash(1, 0x1000193u);
      h1 = order_hash(3, 0x3000193u);
      h2 = order_hash(4, 0x4000193u);
      bit_index = 0;
      partial = 0;
    }
  }
};

struct Encoder {
  std::ostream& out;
  uint32_t x1 = 0, x2 = 0xFFFFFFFFu;
  Encoder(std::ostream& o) : out(o) {}
  void encode_bit(int bit, int p4095) {
    p4095 = clampi(p4095, 1, 4094);
    uint32_t xmid = x1 + (uint32_t)(((uint64_t)(x2 - x1) * (uint32_t)p4095) >> 12);
    if (bit) x2 = xmid;
    else x1 = xmid + 1;
    while (((x1 ^ x2) & 0xFF000000u) == 0) {
      out.put((char)(x2 >> 24));
      x1 <<= 8;
      x2 = (x2 << 8) + 255;
    }
  }
  void flush() {
    for (int i = 0; i < 4; ++i) {
      out.put((char)(x2 >> 24));
      x2 = (x2 << 8) + 255;
    }
  }
};

struct Decoder {
  std::istream& in;
  uint32_t x1 = 0, x2 = 0xFFFFFFFFu, x = 0;
  Decoder(std::istream& i) : in(i) {
    for (int k = 0; k < 4; ++k) {
      int c = in.get();
      x = (x << 8) + (c == EOF ? 0 : (uint8_t)c);
    }
  }
  int decode_bit(int p4095) {
    p4095 = clampi(p4095, 1, 4094);
    uint32_t xmid = x1 + (uint32_t)(((uint64_t)(x2 - x1) * (uint32_t)p4095) >> 12);
    int bit = x <= xmid;
    if (bit) x2 = xmid;
    else x1 = xmid + 1;
    while (((x1 ^ x2) & 0xFF000000u) == 0) {
      x1 <<= 8;
      x2 = (x2 << 8) + 255;
      int c = in.get();
      x = (x << 8) + (c == EOF ? 0 : (uint8_t)c);
    }
    return bit;
  }
};

static void write_u64(std::ostream& o, uint64_t v) {
  for (int i = 7; i >= 0; --i) o.put((char)((v >> (i * 8)) & 0xFF));
}
static uint64_t read_u64(std::istream& i) {
  uint64_t v = 0;
  for (int k = 0; k < 8; ++k) {
    int c = i.get();
    v = (v << 8) | (c == EOF ? 0 : (uint8_t)c);
  }
  return v;
}

static int cmd_compress(const std::string& in_path, const std::string& out_path) {
  std::ifstream in(in_path, std::ios::binary);
  if (!in) { std::cerr << "cannot open " << in_path << "\n"; return 1; }
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), {});
  std::ofstream out(out_path, std::ios::binary);
  if (!out) { std::cerr << "cannot write " << out_path << "\n"; return 1; }
  out.write("CMXB", 4);
  write_u64(out, data.size());
  Predictor pr;
  Encoder enc(out);
  for (uint8_t b : data) {
    for (int k = 0; k < 8; ++k) {
      int bit = (b >> (7 - k)) & 1;
      int p = pr.predict();
      enc.encode_bit(bit, p);
      pr.update(bit);
    }
  }
  enc.flush();
  std::cerr << "in  " << data.size() << " bytes\n";
  out.flush();
  std::cerr << "out " << out.tellp() << " bytes (header+payload)\n";
  return 0;
}

static int cmd_decompress(const std::string& in_path, const std::string& out_path) {
  std::ifstream in(in_path, std::ios::binary);
  if (!in) { std::cerr << "cannot open " << in_path << "\n"; return 1; }
  char mag[4];
  in.read(mag, 4);
  if (std::memcmp(mag, "CMXB", 4) != 0) { std::cerr << "not a CMXB file\n"; return 1; }
  uint64_t n = read_u64(in);
  Predictor pr;
  Decoder dec(in);
  std::ofstream out(out_path, std::ios::binary);
  if (!out) { std::cerr << "cannot write " << out_path << "\n"; return 1; }
  for (uint64_t i = 0; i < n; ++i) {
    int b = 0;
    for (int k = 0; k < 8; ++k) {
      int p = pr.predict();
      int bit = dec.decode_bit(p);
      pr.update(bit);
      b = (b << 1) | bit;
    }
    out.put((char)b);
  }
  std::cerr << "wrote " << n << " bytes\n";
  return 0;
}

static int cmd_demo(const std::string& text) {
  Predictor pr;
  std::cout << "bit  p(1)   A    B    C   D0   D1   D2   wA wB wC ...  ch\n";
  for (unsigned char ch : text) {
    for (int k = 0; k < 8; ++k) {
      int bit = (ch >> (7 - k)) & 1;
      int p = pr.predict();
      std::printf("%d    %4d  %4d %4d %4d %4d %4d %4d  %3d %3d %3d     %c\n",
                  bit, p, pr.last_p[0], pr.last_p[1], pr.last_p[2], pr.last_p[3],
                  pr.last_p[4], pr.last_p[5], pr.mix.w[0], pr.mix.w[1], pr.mix.w[2],
                  (k == 7 && ch >= 32 && ch < 127) ? ch : ' ');
      pr.update(bit);
    }
  }
  std::cout << "final mixer weights:";
  for (int i = 0; i < Mixer::N; ++i) std::cout << " " << pr.mix.w[i];
  std::cout << "\n";
  return 0;
}

static int cmd_generate(int n, const std::string& seed) {
  Predictor pr;
  for (unsigned char ch : seed) {
    for (int k = 0; k < 8; ++k) {
      int bit = (ch >> (7 - k)) & 1;
      pr.predict();
      pr.update(bit);
    }
  }
  std::string out = seed;
  uint32_t rng = 0xC0FFEEu;
  auto rnd = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
  for (int i = 0; i < n; ++i) {
    int b = 0;
    for (int k = 0; k < 8; ++k) {
      int p = pr.predict();
      int bit = ((int)(rnd() % 4095) < p) ? 1 : 0;
      pr.update(bit);
      b = (b << 1) | bit;
    }
    out.push_back((char)b);
  }
  std::cout << out << "\n";
  return 0;
}

static void usage() {
  std::cerr <<
      "cmix-bit -- context-mixing bit predictor prototype\n"
      "\n"
      "  cmix-bit compress   <in> <out.cmxb>\n"
      "  cmix-bit decompress <in.cmxb> <out>\n"
      "  cmix-bit demo       [text]\n"
      "  cmix-bit generate   <nbytes> [seed text]\n";
}

int main(int argc, char** argv) {
  if (argc < 2) { usage(); return 1; }
  std::string cmd = argv[1];
  if (cmd == "compress" && argc == 4) return cmd_compress(argv[2], argv[3]);
  if (cmd == "decompress" && argc == 4) return cmd_decompress(argv[2], argv[3]);
  if (cmd == "demo") {
    std::string t = (argc >= 3) ? argv[2] : "Hello, mixer. 12345 repeated repeated.";
    return cmd_demo(t);
  }
  if (cmd == "generate" && argc >= 3) {
    int n = std::atoi(argv[2]);
    std::string seed = (argc >= 4) ? argv[3] : "The quick brown fox ";
    return cmd_generate(n, seed);
  }
  usage();
  return 1;
}
