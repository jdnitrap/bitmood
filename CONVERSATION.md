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
  **Correction (09-23):** it was a bug. The coder gave bit 1 the (1-p) part
  of the range while p meant P(1), so every bit was charged backwards. Fixed
  in `20b094d`; the same 1800-byte repeated text now codes to ~54 bytes.
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

- `cmix_bit.cpp` / `Makefile` — working CLI prototype (09-23: split into `src/`)
- `docs/pipeline.html` — pipeline drawing
- `docs/mechanical.html` — step-by-step breakdown of each part (describes the
  09-22 prototype; section 6 has the old, inverted coder)

---

# Build-out — 2026-09-23

Goal set by the user: predict and **generate** better; learn only from
bits/bytes, the way patterns show up in ImHex's hex grid (rows, columns,
segment size, word size 8/16/32/64); be told the data type first. CLI only,
no HTML. Screenshots of the idea: `~/Pictures/Screenshots/imhexpic/`
(`carbide_report.html` vs `README.md`, compared on purpose).

## Bugs fixed in the 09-22 prototype

1. Coder inverted (see correction above).
2. Long match took a candidate as best before checking it agreed with the
   bits already decided in the byte; could read before stream start.
3. D0/D1/D2 hashed the whole history, so contexts never recurred.

Later, under ASan/UBSan: zero-length `memcpy` with a null pointer; `write`
dropped keys typed while a memory loaded (`TCSAFLUSH`).

## What exists now (one commit each)

| Phase | What |
|---|---|
| 1 | `src/` modules; Model (learned) / Stream (context) / BitPos split; saved memory (`train --state`, checksummed, atomic); `generate` never learns its own output |
| 1.5 | Grid specialist E (the ImHex idea): line rows + 2 auto widths found by scoring distances 2..1024; view tracker picks mixer weight sets; `compare` |
| 2 | Whole-byte sampling from the 255-node bit tree; `--charset seen`; `--novelty N` (suffix array + own-output hashes) |
| 3 | `--blend` (mix / product); `--line-start`, `--acrostic`, `--max-line`, `--words`, `--rhyme`; `--best-of N` |
| 4 | `write`: terminal co-writing, grey suggestion, Tab / → / Backspace, learns finished words |
| 5 | Shared context table with collision checks; orders 0–8; word and learned byte-class specialists; hash-indexed long match (16 MB); 3 float mixers + final + 2 APMs; 16-bit coder |
| types | `--type image` (PGM/PPM), `audio` (16-bit WAV), `raw`; specialists I and S; per-type views; `.ppm`/`.wav` output |
| level 2 | Optional LSTM specialist L (`--lstm N`) |

`make test` runs 54 end-to-end checks (round trips, memory identity,
corruption, generation rules, pseudo-terminal `write`, types, LSTM).

## Measured

- 300 KB mix (licenses, Python, HTML, markdown): 63.3 KB vs `xz -9e` 79.1 KB.
- `carbide_report.html`: 2.73 → 2.29 bits/byte; the grid saves 7% (it finds
  the 29-byte chart records `[24844, 1.2093673664696363], `). `README.md`:
  3.38 → 2.87; the grid is neutral on prose.
- Type label alone: RGB picture 17.1 KB (`image`) vs 21.4 KB (`raw`); tone
  22.9 KB (`audio`) vs 26.7 KB (`raw`).
- `compare --type raw` on the cmix-bit binary finds the 24-byte ELF symbol
  records and 64-bit pointer regions without being told the format.
- LSTM: 0.3–1.3% smaller at ~7–8× the time, so opt-in.

## Known limits

- Still a pattern model: words, phrases and style, not meaning or plot.
- Tiny or very repetitive memories + low `--novelty` → noise (the rule
  forbids everything the model believes). Raise `--novelty`.
- Rhyme is rough (last two letters) and needs `--best-of` to be reliable.
- Generated audio is mostly noise shaped by the training tone.
- Learning one file slightly hurts the next when their structure differs
  (cross-file +0.4–1.6% on the two carbide files).
- Memories are ~36 MB by default (`--table-bits 18` → ~6 MB).
- Floats in the mixers: compressed files decode with the same build on
  the same kind of machine; not guaranteed across compilers/CPUs.
