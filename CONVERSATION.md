# Conversation notes — 2026-09-22

Working session on this repo: what GitHub Projects are, a C++ prototype,
how the model learns, how to generate text without a transformer, how to
keep state, and what a graph specialist would add.

## Language

C++ is the default for the engine (bit loop, tables, coder). Python is
fine as a lab to inspect mixer weights. Prototype is C++: `cmix_bit.cpp`.

## Prototype tests

Built with `g++ -std=c++17 -O2`. CLI: compress, decompress, demo, generate.

- 56-byte string compress → decompress: byte-identical.
- 1360-byte repeated text: byte-identical.
- Sizes grew (56→91, 1360→14813). Coder + cold tables, not a broken invert.
- Demo: first bits ~2048; after a letter, specialist C moves; late in
  "repeated" mixed p swings ~200 vs ~3800. Final mixer weights about
  `172 106 194 128 128 136`.
- Generate from a short seed: mostly noise until tables are warmed.

## Learn vs retain

Learns online every bit (n0/n1, mixer weights, B window). Does **not**
retain after process exit. `.cmxb` is coded payload, not a saved brain.
Retention = serialize A/D tables + mixer weights + hashes (`state.bin`),
or persist a graph. Decode rebuilds tables by replaying the file; that is
lockstep, not a reusable writer.

No separate fine-tune stage. Adaptation is the bit loop. Better ratios
need design changes (coder, decay, match finder), not epochs.

## Generative writer (same model, no transformer)

Same predict → sample → update loop.

- Warm on a corpus first (update only).
- Temperature on stretch/squash.
- Reject non-printable bytes and resample eight bits.
- Bias mixer: raise B to copy, keep C for letter/digit shape, cap B to avoid lock-in.
- Chunk drafts: generate, score, append keeper, continue.
- Rider UI: emit characters, tab-complete next byte, update() on accept or typed byte.

Creative here means warmer tables + temperature + specialist knobs +
reject/resample + iteration. Not meaning or plot.

## Graph as extra specialist

Node = context (hash, n-gram, named flag row). Edge = “next bit/byte”
with strength. Walk from hist; emit `p_G`; mixer gets a seventh vote;
update the edge after the real bit. Save nodes/edges to keep learning
across days. Search-on-miss; candidate needs a yes before it is a fact.

## Second prototype (still no transformer)

Replace FPAQ0-style coder with a tighter range/ANS coder. Add count
decay. Put bit_index in hash keys cleanly. Replace linear 64 KiB scan
with rolling hash / suffix structure.

## Positives

Interpretable votes, online per file, CPU-only, add a specialist without
rewiring the loop, bit-exact invert when coder and updates match.

## Negatives

Does not shrink files yet. Shallow memory. No default checkpoint.
Cold generate is junk. No semantics. Sparse high-order tables.
Sequential bit loop. Domain shift hurts. One wrong coded bit desyncs
decode. Constants are hand-tuned.

## Files from this session

- `cmix_bit.cpp` / `Makefile` — working CLI prototype
- `docs/pipeline.html` — pipeline drawing
- `docs/mechanical.html` — step-by-step breakdown of each part
