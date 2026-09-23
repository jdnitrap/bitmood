#include "io/formats.h"

#include <cctype>
#include <cstring>
#include <stdexcept>

#include "io/files.h"

namespace cmix {

namespace {

// ---- PNM (P5 / P6) ------------------------------------------------------------

bool parse_pnm(const std::vector<uint8_t>& f, TypedData& d) {
  if (f.size() < 3 || f[0] != 'P' || (f[1] != '5' && f[1] != '6')) return false;
  size_t i = 2;
  auto skip = [&]() {
    for (;;) {
      while (i < f.size() && std::isspace(f[i])) ++i;
      if (i < f.size() && f[i] == '#') {
        while (i < f.size() && f[i] != '\n') ++i;
      } else {
        return;
      }
    }
  };
  auto number = [&]() {
    skip();
    long v = 0;
    size_t start = i;
    while (i < f.size() && std::isdigit(f[i]) && v < 1000000) v = v * 10 + (f[i++] - '0');
    if (i == start) throw std::runtime_error("PNM header: expected a number");
    return v;
  };
  const long w = number(), h = number(), maxval = number();
  if (i >= f.size() || !std::isspace(f[i])) throw std::runtime_error("PNM header: missing whitespace before data");
  ++i;
  if (maxval != 255) throw std::runtime_error("only 8-bit PNM images (maxval 255) are supported");
  if (w <= 0 || h <= 0) throw std::runtime_error("PNM header: bad size");
  d.width = (int)w;
  d.height = (int)h;
  d.channels = f[1] == '5' ? 1 : 3;
  d.header.assign(f.begin(), f.begin() + (std::ptrdiff_t)i);
  d.payload.assign(f.begin() + (std::ptrdiff_t)i, f.end());
  return true;
}

// ---- WAV ----------------------------------------------------------------------

uint32_t le32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
uint16_t le16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

void parse_wav(const std::vector<uint8_t>& f, TypedData& d) {
  if (f.size() < 12 || std::memcmp(f.data(), "RIFF", 4) != 0 || std::memcmp(f.data() + 8, "WAVE", 4) != 0)
    throw std::runtime_error("not a WAV file");
  size_t i = 12;
  bool have_fmt = false;
  while (i + 8 <= f.size()) {
    const uint32_t len = le32(&f[i + 4]);
    const uint8_t* body = &f[i + 8];
    if (std::memcmp(&f[i], "fmt ", 4) == 0) {
      if (len < 16 || i + 8 + 16 > f.size()) throw std::runtime_error("WAV: short fmt chunk");
      const uint16_t format = le16(body), bits = le16(body + 14);
      if (format != 1 || bits != 16) throw std::runtime_error("only 16-bit PCM WAV is supported");
      d.channels = le16(body + 2);
      d.sample_rate = (int)le32(body + 4);
      if (d.channels < 1 || d.channels > 2) throw std::runtime_error("only mono or stereo WAV is supported");
      have_fmt = true;
    } else if (std::memcmp(&f[i], "data", 4) == 0) {
      if (!have_fmt) throw std::runtime_error("WAV: data before fmt");
      d.header.assign(f.begin(), f.begin() + (std::ptrdiff_t)(i + 8));
      d.payload.assign(f.begin() + (std::ptrdiff_t)(i + 8), f.end());
      return;
    }
    i += 8 + len + (len & 1);
  }
  throw std::runtime_error("WAV: no data chunk");
}

void put32(std::vector<uint8_t>& o, uint32_t v) {
  for (int k = 0; k < 4; ++k) o.push_back((uint8_t)(v >> (8 * k)));
}
void put16(std::vector<uint8_t>& o, uint16_t v) {
  o.push_back((uint8_t)v);
  o.push_back((uint8_t)(v >> 8));
}

}  // namespace

TypedData read_typed(const std::string& path, DataType t, bool raw_image_ok) {
  std::vector<uint8_t> f = read_file(path);
  TypedData d;
  try {
    switch (t) {
      case DataType::Text:
      case DataType::Raw: d.payload = std::move(f); break;
      case DataType::Image:
        if (!parse_pnm(f, d)) {
          if (!raw_image_ok) throw std::runtime_error("not a binary PGM/PPM image (P5/P6)");
          d.payload = std::move(f);  // raw pixels; shape comes from --width/--channels
        }
        break;
      case DataType::Audio: parse_wav(f, d); break;
    }
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
  return d;
}

void apply_shape(Config& cfg, const TypedData& d, bool creating, const std::string& what) {
  if (cfg.type == DataType::Image && d.width > 0) {
    if (creating && cfg.width == 0) {
      cfg.width = d.width;
      cfg.channels = d.channels;
    } else if (d.width != cfg.width || d.channels != cfg.channels) {
      throw std::runtime_error(what + " is " + std::to_string(d.width) + " px wide with " +
                               std::to_string(d.channels) + " channel(s); this memory is for " +
                               std::to_string(cfg.width) + " px, " + std::to_string(cfg.channels));
    }
  }
  if (cfg.type == DataType::Audio) {
    if (creating && cfg.channels == 0) {
      cfg.channels = d.channels;
      cfg.sample_rate = d.sample_rate;
    } else if (d.channels != cfg.channels) {
      throw std::runtime_error(what + " has " + std::to_string(d.channels) + " channel(s); this memory is for " +
                               std::to_string(cfg.channels));
    }
  }
  if (cfg.type == DataType::Image && (cfg.width <= 0 || (cfg.channels != 1 && cfg.channels != 3)))
    throw std::runtime_error("image memory needs a PGM/PPM file, or --width and --channels 1|3 for raw pixels");
}

std::vector<uint8_t> make_container(const Config& cfg, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> out;
  if (cfg.type == DataType::Image) {
    const size_t row = (size_t)cfg.width * (size_t)cfg.channels;
    const size_t h = row ? payload.size() / row : 0;
    std::string head = std::string(cfg.channels == 1 ? "P5" : "P6") + "\n" + std::to_string(cfg.width) + " " +
                       std::to_string(h) + "\n255\n";
    out.assign(head.begin(), head.end());
    out.insert(out.end(), payload.begin(), payload.begin() + (std::ptrdiff_t)(h * row));
  } else if (cfg.type == DataType::Audio) {
    const uint16_t ch = (uint16_t)cfg.channels;
    const size_t frame = 2u * ch;
    const size_t n = payload.size() / frame * frame;
    out = {'R', 'I', 'F', 'F'};
    put32(out, (uint32_t)(36 + n));
    for (char c : std::string("WAVEfmt ")) out.push_back((uint8_t)c);
    put32(out, 16);
    put16(out, 1);
    put16(out, ch);
    put32(out, (uint32_t)cfg.sample_rate);
    put32(out, (uint32_t)(cfg.sample_rate * frame));
    put16(out, (uint16_t)frame);
    put16(out, 16);
    for (char c : std::string("data")) out.push_back((uint8_t)c);
    put32(out, (uint32_t)n);
    out.insert(out.end(), payload.begin(), payload.begin() + (std::ptrdiff_t)n);
  } else {
    out = payload;
  }
  return out;
}

}  // namespace cmix
