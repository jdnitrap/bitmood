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

`make test` runs 86 end-to-end checks (round trips, memory identity,
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

## Graph specialist (added after the build-out)

The 09-22 graph idea, built for three data types (`--graph`, opt-in):
nodes are words (text), 20 ms slice shapes (audio) or 8-pixel run shapes
(image); order-1 and order-2 edges; an edge votes only after
`--graph-confirm` sightings (default 2). `cmix-bit graph` shows it.
`generate --graph-plan` picks the next unit from the graph and steers
toward it (`--plan-strength`, defaults text 4 / audio 100 / image 2;
`--plan-temp`, default 1).

Measured:
- Text: known word pairs 51.5% -> 64.7% (strength 4) / 71.8% (20); real
  words 91% -> 92-94%. Compression gain < 0.1% (orders 5-8 already
  cover word transitions).
- Audio (repeating melody): without planning, generation falls silent
  (0-6% loud slices vs 78% in training). Planning doubles to triples the
  loud share, but the notes stay noisy. Two fixes were needed on the way:
  plans get their own temperature (squaring weights kept "stay quiet"
  winning), and steering is (1 + strength)^preference (a linear boost
  capped the effect at ~3x).
- Image: planning flattens colour and adds bands; not recommended.

## SNN on the graph (design: `~/stuff.txt`)

Graph nodes as neurons, edges as synapses (`--graph --snn`). A finished
token spikes, charge spreads along edges (weight x count share) and leaks
(x0.6 per token step); the charged neurons are vote N and the source of
plans. Three-factor learning: eligibility = neurons fired before the unit;
modulation = the unit's surprise against its running average ("dopamine");
fired -> predicted synapses move by rate x trace x d; when surprised,
fired -> actual is strengthened. Readout (text): learned gain and offset.
Change detection: fast (8 units) vs slow (1024 units) surprise averages;
a jump (fast > 2 x slow + 1) opens a 128-unit "new region" with its own
mixer weight set; surprise is also checked every 32 bytes without a
finished unit.

Measured:
- N alone 5.81 bits/byte on licenses vs G 6.83 (after learning; 6.62
  before). README 2.882 -> 2.846 overall; joined 4-file test 62,551 ->
  62,521 bytes.
- Planned text: real words 93.2% (G 92.5%), known word pairs 55.4% (G 64.7%).
- Audio melody at temperature 1: loud slices 61% vs 47% with the plain
  graph (training 78%); notes still noisy.
- Images: slightly more colour; compression unchanged.
- Change detection: file joins found within 54-176 bytes, plus a join
  inside one file and CSS -> JavaScript; misses changes that don't raise
  surprise (text -> random digits). Looser thresholds gave 49-116 false
  alarms on the same test.

## Training on 1 / 10 / 50 MB (Project Gutenberg novels)

61 public-domain novels (51 MB after stripping headers); three held out
(Jekyll and Hyde, Heart of Darkness, The Yellow Wallpaper, 377 KB).

| | 1 MB | 10 MB | 50 MB |
|---|---|---|---|
| held-out bits/byte, plain | 1.963 | 1.884 | 1.870 |
| held-out, graph + SNN | 1.964 | 1.879 | 1.857 |
| plain, real words / known pairs | 95.7 / 69.5% | 96.9 / 78.1% | 97.5 / 85.2% |
| graph + SNN + plan | 97.2 / 73.9% | 98.0 / 85.4% | 99.0 / 89.6% |
| plain: time, RAM, memory | 6 s, 170, 37 MB | 57 s, 230, 46 MB | 293 s, 220, 46 MB |
| graph + SNN (no cap): time, RAM, memory | 9 s, 194, 41 MB | 111 s, 372, 73 MB | 621 s, 763, 138 MB |

Findings and fixes:
- The default 32 MB table saturates after ~10 MB: `--table-bits 24` on
  50 MB plain gives 1.780 held-out (5% better; 142 MB memory, 604 MB RAM).
- The graph grew without limit (2.1 M nodes at 50 MB). Now capped at
  `--graph-max-nodes` (default 524,288), pruning never-confirmed, weakest
  nodes first: 50 MB graph + SNN -> 470 MB RAM, 96 MB memory, held-out
  still 1.857.
- Planning produced "the the" / "of the the" (planned common words). Plans
  now skip the last two words and weight candidates by count / sqrt(word
  frequency): doubled words 2.6% -> 0.3-0.4%. The known-pair score drops
  (85 -> 82-86%), because it rewarded exactly those common pairs; the text
  reads better.

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
