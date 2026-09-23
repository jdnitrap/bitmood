#include "model/lstm.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/math.h"

namespace cmix {

namespace {

constexpr float kLr = 0.002f;
constexpr float kBeta1 = 0.9f, kBeta2 = 0.999f, kEps = 1e-8f;
constexpr float kClip = 1.0f;

inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

void Lstm::Param::init(size_t n, float scale, uint32_t& seed) {
  w.resize(n);
  for (float& x : w) {
    seed = seed * 1664525u + 1013904223u;  // fixed seed: every new model starts the same
    x = scale * ((float)(seed >> 8) / 8388608.0f - 1.0f);
  }
  resize_like();
}

Lstm::Lstm(int cells) : h_(cells) {
  if (cells < 1 || cells > LstmState::kMaxCells) throw std::runtime_error("--lstm cells must be 1..64");
  uint32_t seed = 12345;
  const size_t g = 4 * (size_t)h_;
  wx_.init(256 * g, 0.1f, seed);
  wh_.init(g * (size_t)h_, 1.0f / std::sqrt((float)h_), seed);
  b_.init(g, 0.0f, seed);
  for (int j = 0; j < h_; ++j) b_.w[(size_t)(h_ + j)] = 1.0f;  // forget gate starts open
  v_.init(256 * (size_t)h_, 0.1f, seed);
  c_.init(256, 0.0f, seed);
  ring_.resize(kHorizon);
  for (Step& s : ring_)
    for (auto* v : {&s.i, &s.f, &s.o, &s.g, &s.c_prev, &s.h_prev, &s.c, &s.dh}) v->assign((size_t)h_, 0.0f);
}

void Lstm::forward(LstmState& s, int b) const {
  const int H = h_;
  std::copy(s.h, s.h + H, s.h_prev);
  std::copy(s.c, s.c + H, s.c_prev);
  const float* wx = &wx_.w[(size_t)b * 4 * H];
  for (int r = 0; r < 4 * H; ++r) {
    float a = wx[r] + b_.w[(size_t)r];
    const float* wh = &wh_.w[(size_t)r * H];
    for (int k = 0; k < H; ++k) a += wh[k] * s.h_prev[k];
    const int gate = r / H, j = r % H;
    if (gate == 0) s.gi[j] = sigm(a);
    else if (gate == 1) s.gf[j] = sigm(a);
    else if (gate == 2) s.go[j] = sigm(a);
    else s.gg[j] = std::tanh(a);
  }
  for (int j = 0; j < H; ++j) {
    s.c[j] = clampf(s.gf[j] * s.c_prev[j] + s.gi[j] * s.gg[j], -50.0f, 50.0f);
    s.h[j] = s.go[j] * std::tanh(s.c[j]);
  }
  // Next-byte distribution (softmax), stored as a prefix-sum tree.
  float logit[256], top = -1e30f;
  for (int k = 0; k < 256; ++k) {
    float a = c_.w[(size_t)k];
    const float* v = &v_.w[(size_t)k * H];
    for (int j = 0; j < H; ++j) a += v[j] * s.h[j];
    logit[k] = a;
    top = std::max(top, a);
  }
  float sum = 0;
  for (int k = 0; k < 256; ++k) sum += (logit[k] = std::exp(logit[k] - top));
  for (int k = 0; k < 256; ++k) s.tree[256 + k] = std::max(logit[k] / sum, 1e-6f);
  for (int n = 255; n >= 1; --n) s.tree[n] = s.tree[2 * n] + s.tree[2 * n + 1];
  s.x = b;
  s.ready = true;
}

int Lstm::p_bit(const LstmState& s, int bit_index, int partial) const {
  if (!s.ready) return 32768;
  const int id = (1 << bit_index) | partial;
  return to_p16(clampf(s.tree[2 * id + 1] / s.tree[id], 1e-4f, 1.0f - 1e-4f));
}

void Lstm::adam(Param& p, float lr) {
  const float c1 = 1.0f - std::pow(kBeta1, (float)t_), c2 = 1.0f - std::pow(kBeta2, (float)t_);
  for (size_t i = 0; i < p.w.size(); ++i) {
    const float g = clampf(p.g[i], -kClip, kClip);
    p.m[i] = kBeta1 * p.m[i] + (1 - kBeta1) * g;
    p.v[i] = kBeta2 * p.v[i] + (1 - kBeta2) * g * g;
    p.w[i] -= lr * (p.m[i] / c1) / (std::sqrt(p.v[i] / c2) + kEps);
    p.g[i] = 0;
  }
}

void Lstm::learn(const LstmState& s, int b) {
  if (!s.ready) return;
  const int H = h_;
  ++t_;
  // Output layer: softmax cross-entropy gradient.
  float dh[LstmState::kMaxCells] = {};
  const float total = s.tree[1];
  for (int k = 0; k < 256; ++k) {
    const float d = s.tree[256 + k] / total - (k == b ? 1.0f : 0.0f);
    float* vg = &v_.g[(size_t)k * H];
    const float* vw = &v_.w[(size_t)k * H];
    for (int j = 0; j < H; ++j) {
      vg[j] += d * s.h[j];
      dh[j] += vw[j] * d;
    }
    c_.g[(size_t)k] += d;
  }
  adam(v_, kLr);
  adam(c_, kLr);
  // Keep the step for backpropagation through time.
  if (s.x < 0) return;
  Step& st = ring_[steps_];
  st.x = s.x;
  std::copy(s.gi, s.gi + H, st.i.begin());
  std::copy(s.gf, s.gf + H, st.f.begin());
  std::copy(s.go, s.go + H, st.o.begin());
  std::copy(s.gg, s.gg + H, st.g.begin());
  std::copy(s.c_prev, s.c_prev + H, st.c_prev.begin());
  std::copy(s.h_prev, s.h_prev + H, st.h_prev.begin());
  std::copy(s.c, s.c + H, st.c.begin());
  std::copy(dh, dh + H, st.dh.begin());
  if (++steps_ == (uint32_t)kHorizon) {
    bptt();
    steps_ = 0;
  }
}

void Lstm::bptt() {
  const int H = h_;
  std::vector<float> dh_next((size_t)H, 0.0f), dc_next((size_t)H, 0.0f), da((size_t)(4 * H));
  for (int k = kHorizon - 1; k >= 0; --k) {
    const Step& st = ring_[(size_t)k];
    for (int j = 0; j < H; ++j) {
      const float dh = st.dh[(size_t)j] + dh_next[(size_t)j];
      const float tc = std::tanh(st.c[(size_t)j]);
      const float i = st.i[(size_t)j], f = st.f[(size_t)j], o = st.o[(size_t)j], g = st.g[(size_t)j];
      const float dc = dh * o * (1 - tc * tc) + dc_next[(size_t)j];
      da[(size_t)j] = dc * g * i * (1 - i);
      da[(size_t)(H + j)] = dc * st.c_prev[(size_t)j] * f * (1 - f);
      da[(size_t)(2 * H + j)] = dh * tc * o * (1 - o);
      da[(size_t)(3 * H + j)] = dc * i * (1 - g * g);
      dc_next[(size_t)j] = dc * f;
    }
    std::fill(dh_next.begin(), dh_next.end(), 0.0f);
    float* wxg = &wx_.g[(size_t)st.x * 4 * H];
    for (int r = 0; r < 4 * H; ++r) {
      const float a = da[(size_t)r];
      wxg[r] += a;
      b_.g[(size_t)r] += a;
      float* whg = &wh_.g[(size_t)r * H];
      const float* whw = &wh_.w[(size_t)r * H];
      for (int k2 = 0; k2 < H; ++k2) {
        whg[k2] += a * st.h_prev[(size_t)k2];
        dh_next[(size_t)k2] += whw[k2] * a;
      }
    }
  }
  adam(wx_, kLr);
  adam(wh_, kLr);
  adam(b_, kLr);
}

void Lstm::save(Writer& w) const {
  w.i32(h_);
  for (const Param* p : {&wx_, &wh_, &b_, &v_, &c_}) {
    w.vec_f32(p->w);
    w.vec_f32(p->m);
    w.vec_f32(p->v);
  }
  for (const Step& s : ring_) {
    w.i32(s.x);
    for (const auto* v : {&s.i, &s.f, &s.o, &s.g, &s.c_prev, &s.h_prev, &s.c, &s.dh}) w.vec_f32(*v);
  }
  w.u32(steps_);
  w.u64(t_);
}

void Lstm::load(Reader& r) {
  if (r.i32() != h_) throw std::runtime_error("state file: LSTM size mismatch");
  for (Param* p : {&wx_, &wh_, &b_, &v_, &c_}) {
    const size_t n = p->w.size();
    r.vec_f32(p->w);
    r.vec_f32(p->m);
    r.vec_f32(p->v);
    if (p->w.size() != n || p->m.size() != n || p->v.size() != n) throw std::runtime_error("state file: LSTM size mismatch");
  }
  for (Step& s : ring_) {
    s.x = r.i32();
    for (auto* v : {&s.i, &s.f, &s.o, &s.g, &s.c_prev, &s.h_prev, &s.c, &s.dh}) {
      r.vec_f32(*v);
      if (v->size() != (size_t)h_) throw std::runtime_error("state file: LSTM size mismatch");
    }
  }
  steps_ = r.u32();
  t_ = r.u64();
  if (steps_ >= (uint32_t)kHorizon) throw std::runtime_error("state file: bad LSTM step count");
}

}  // namespace cmix
