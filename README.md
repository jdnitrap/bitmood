# Context-Mixing Bit Predictor

**One-line idea:** instead of one big model guessing the next byte, use a handful
of small, simple, *specialized* guessers — each one an expert in one kind of
pattern — and let a tiny "referee" combine their opinions into a single
probability for the next **bit**.

This is not a new invention. It's the core idea behind the best lossless
compressors in the world (PAQ, cmix, zpaq). Those tools compress better than
zip/gzip largely *because* of this trick, not because of a bigger neural net.

**Session notes:** see [CONVERSATION.md](CONVERSATION.md) — 2026-09-22: C++ prototype,
learn vs retain, generative writer without a transformer, graph specialist;
2026-09-23: the build-out (bug fixes, phases 1–5, ImHex grid, data types, LSTM,
measurements and known limits).

Build the prototype:

```
make
make test                                   # end-to-end checks
./cmix-bit demo "Hello mixer"
./cmix-bit compress in.txt out.cmxb
./cmix-bit decompress out.cmxb out.txt
```

Saved memory and generation:

```
./cmix-bit train --state brain.bin notes.txt more.txt    # learns; keeps learning if brain.bin exists
./cmix-bit info brain.bin
./cmix-bit generate 300 "The " --state brain.bin --temp 0.8 --seed 42
```

`generate` never learns from its own output and never changes the memory
file. With `--state` the prompt only sets the context; without it the prompt
is the only thing the model learns from.

Each byte is chosen as a whole: the model's bit predictions at all 255
nodes of the byte's bit tree give a probability for each of the 256
bytes, then filters and sampling settings apply:

| Option | Effect |
|---|---|
| `--temp T` | < 1 safer and more repetitive, > 1 wilder (default 1) |
| `--top-k K`, `--top-p P` | keep only the K most likely bytes / the most likely bytes covering P of the probability |
| `--charset seen` | (default) valid UTF-8 using only bytes that occur in the training text; also `utf8`, `ascii`, `any` |
| `--novelty N` | never copy more than N bytes in a row from the training text *or* from its own earlier output (stops loops) |
| `--stats` | print model surprise (bits/byte) and the longest run copied from the training text |

With a small or very repetitive memory, a low `--novelty` forbids almost
everything the model believes and the output turns to noise; raise it.

Creativity controls:

| Option | Effect |
|---|---|
| `--state a.bin --state b.bin --blend 0.7,0.3` | write from several memories at once |
| `--blend-mode mix` | (default) weighted average of the memories' byte probabilities: each keeps its confident choices, so styles alternate |
| `--blend-mode product` | average in stretch space per bit: only what the memories agree on stays likely |
| `--line-start CHARS` | every line starts with one of CHARS |
| `--acrostic WORD` | line *i* starts with letter *i* of WORD |
| `--max-line N` | lines end by column N |
| `--words FILE` | only words from FILE (between words: spaces and ordinary punctuation) |
| `--rhyme` | rough AABB rhyme: the second line of a pair ends as soon as it finishes a word ending like the first line's; with `--best-of N`, rhyming lines are preferred |
| `--best-of N` | write N candidate lines, keep the one whose surprise is closest to real text's, with no nonsense spikes, no long copies, and a length like the training text's lines |

Soft rules are relaxed in order when they conflict: novelty first, then
rhyme, then line length. Charset, line start, acrostic and word list never are.

Writing together with the model (in a terminal):

```
./cmix-bit write --state brain.bin --out draft.txt
```

The model's guess for what comes next appears in grey after the cursor.
**Tab** accepts it, **→** accepts one word, **Backspace** edits the word
you are on, **Ctrl-S** saves, **Ctrl-D** or **Esc** saves and quits. Each
finished word (at a space, punctuation, Enter or an accepted suggestion)
is learned, so the memory picks up your writing as you go.

Grid views (the ImHex idea) and file comparison:

```
./cmix-bit compare carbide_report.html README.md
```

A hex editor shows a file as rows and columns; set the column count right
and repeating records line up as vertical stripes. Specialist **E** does
this for the model, in several *views* at once: rows cut at newlines, and
two row widths the model finds itself by scoring every distance 2..1024
("how often does a byte equal the byte d back?"). Each view votes with the
byte *above* the one being predicted. A tracker follows which view is
predicting best right now, and each winning view gets its own mixer weight
set, so different regions of a file learn different trust. `compare` shows
bits per byte with and without the grid, each specialist alone, the
segments where each view won, and whether learning one file helps the other.

### Data types

The model always learns from bytes; `--type` tells it what the bytes are,
which decides where it looks first and keeps each kind in its own memory:

| Type | Files | Extra specialist | Grid views |
|---|---|---|---|
| `text` (default) | anything | W (words) | line rows, 2 auto widths |
| `image` | binary PGM/PPM (P5/P6, 8-bit), or raw pixels with `--width W --channels 1\|3` | I1–I6: left, above, their average, gradient, neighbourhood, colour/vertical trend | image row, one pixel, auto width |
| `audio` | 16-bit PCM WAV, mono or stereo | S1–S4: last sample, linear and curve trends, level (low/high byte aware) | 16-bit sample, frame, auto width |
| `raw` | anything | none | line rows, 2 auto widths, 16/32/64-bit words |

```
./cmix-bit train --state pics.bin --type image a.ppm b.ppm
./cmix-bit generate --state pics.bin --height 96 --out new.ppm --temp 0.7
./cmix-bit train --state sound.bin --type audio tone.wav
./cmix-bit generate --state sound.bin --seconds 2 --out new.wav --temp 0.5
./cmix-bit compress --type image a.ppm a.cmxb          # header kept, pixels modelled
./cmix-bit compare --type raw prog_a prog_b
```

The file header (PNM/WAV) is set aside; only pixels or samples are learned,
and each file starts at pixel/sample 0. On test data the type label alone
makes a picture 20% smaller and a sound 14% smaller than `--type raw`.
Generated images keep the training pictures' palette and smooth regions;
generated audio is the weakest output (mostly noise shaped by the tone).
On a compiled program, `compare --type raw` finds the 24-byte ELF symbol
records and 64-bit pointer regions on its own.

### The engine

| Specialist | Looks at |
|---|---|
| O0, D1, **A** (order 2), D3, D4, D5, D6, D8 | the last 0–8 bytes |
| I1–I6, S1–S4 | neighbouring pixels (image), earlier samples (audio) |
| W1, W2 | the word in progress; it plus the previous word (text) |
| C1, C2 | byte classes of the last bytes (the MDBE flags: letter, digit, space, punctuation, UTF-8…), learned, plus column |
| E | grid views (above) |
| B1, B2 | long match: the most recent earlier place the last 5+ bytes occurred, found with a hash index over up to 16 MB of history |
| L (optional) | level 2: a small byte-level LSTM (`--lstm N`, 1–64 cells) |

Context statistics live in one shared table (`--table-bits`, default 22 =
32 MB) of adaptive probabilities with 16-bit collision checks in buckets of
four. Three mixers with float weights (negative allowed) combine the votes,
each with its own weight sets chosen by: match state × bit position,
previous byte, winning grid view × bit position. A final mixer blends
those, and two APM stages (order 0, order 1) refine the result. The coder
works with 16-bit probabilities. On a 300 KB mix of licenses, Python source,
HTML and markdown this is about 20% smaller than `xz -9e`.

A memory file holds the table, the history, the mixers and the APMs: about
36 MB by default, about 6 MB with `--table-bits 18` (fine for a few hundred
KB of training text).

### Level 2: LSTM specialist

```
./cmix-bit train --state brain.bin --lstm 32 corpus.txt
./cmix-bit compress --lstm 32 in.txt out.cmxb
```

Specialist **L** is a small LSTM (the recurrent network cmix uses, not a
transformer; CPU only). It reads one byte at a time and keeps a running
memory in its hidden state, giving a probability for each next byte that
the mixer takes as one more vote. It learns online: the output layer every
byte, backpropagation through the last 16 bytes every 16 bytes (Adam). It
is off by default because it costs about 7-8x the time; on the test files
it saves 0.3-1.3%, and it matters more the more text it trains on. The
LSTM's hidden state is part of the stream, so generation, rollback and
"train in one run == train in several" all work with it.

---

## 1. The plain-English version

Imagine you're trying to guess the next character in a sentence, one bit at a
time (a bit is just a 0 or a 1 — 8 bits make one byte, one byte is usually one
character). Instead of asking one all-knowing oracle "what's next?", you ask
several narrow specialists at the same time:

- **Specialist A ("recent pattern"):** "In the last few times I saw this exact
  sequence of bytes, what came right after?" (a short-term memory / lookup)
- **Specialist B ("long match"):** "Have I seen this *exact* long stretch of
  text before somewhere earlier? If so, just predict a repeat of what
  followed it last time." (great for repeated words, boilerplate, code)
- **Specialist C ("byte-shape rules"):** "Is the current byte a letter? A
  digit? Mid-word? Then the next bit is *probably* going to keep us inside
  that same kind of byte." (this is exactly what MDBE's `is_alpha` /
  `is_digit` / `is_punct` flags already are!)
- **Specialist D ("order-N statistics"):** "Out of all bytes I've ever seen
  following these last N bytes, here's the frequency table." (classic
  Markov-chain style counting)

Each specialist is cheap — mostly just counters and small lookup tables, not
a heavy neural network. None of them is very smart alone. But a small
**mixer** (a tiny logistic-regression-style model, or a very small neural
net) learns how much to *trust* each specialist right now, and blends their
guesses into one final probability: *"70% chance the next bit is 1."*

That single probability is all you need to either:
- **generate** the next bit (sample from it), or
- **compress** the data (feed it to an arithmetic coder, which uses the
  probability to spend fewer bits on predictable data and more bits on
  surprising data — this is literally how PAQ/cmix achieve state-of-the-art
  compression ratios).

---

## 2. Why this fits your MDBE project so well

You already have this shape of idea, just not connected to *bit prediction*
yet. Look at how MDBE builds a row for each byte:

```
[ learned embedding dims ] [ 6 hard flags ] [ grammar/mechanics scores ]
        (Specialist ?)      (Specialist C)      (Specialist A/D-ish)
```

MDBE's "beside, not inside" rule — named columns sitting next to the learned
embedding instead of blended into it — is *already* a mixing philosophy. A
context-mixing bit predictor is the same idea taken one step further: each
named column becomes its own small, independent predictor of the next bit,
and a mixer (which could be as small as a single trained layer) learns how
much weight to give each one, live, per byte.

Concretely, "Specialist C" above already exists in your code:
`mdbe_constraints()` (the 6 flags: `is_alpha`, `is_digit`, `is_upper`,
`is_punct`, `is_space`, `utf8_lead`) is a perfect example of a free,
always-correct specialist. You'd be adding a few more specialists next to it
and a small mixer to combine them, all working at the level of individual
bits instead of whole 256-way byte choices.

---

## 3. Step by step: how the numbers actually flow

1. You're about to predict bit `k` of the current byte (0 = most significant
   bit, ... 7 = least significant bit).
2. Every specialist outputs one number: its own guess for
   `P(bit_k = 1)` — a probability between 0 and 1.
3. Each probability gets converted to "stretch" space (a stretch function,
   `ln(p / (1-p))`, turns a 0–1 probability into a plain number that's easy
   to add and average — this is the standard trick used in real bit-level
   compressors).
4. The mixer is just a weighted sum of the stretched specialist outputs,
   squashed back into a 0–1 probability (a `squash` function, the inverse of
   stretch — this whole stretch → weighted sum → squash pattern is exactly
   logistic regression).
5. The mixer's weights are updated a tiny bit after every real bit is seen
   (an online learning rule — "if a specialist was right, trust it slightly
   more next time; if it was wrong, trust it slightly less").
6. Repeat for all 8 bits to get a full byte.

None of this requires a GPU. Real compressors like this run fast on a single
CPU core because every specialist is small and cheap.

---

## 4. Strengths

- **Fast and cheap.** No large matrix multiplies needed for the specialists
  themselves — mostly counters, hash tables, and small dot products.
- **Interpretable.** You can always ask "which specialist is currently being
  trusted the most?" — the mixer's weights tell you directly. This matches
  your project's existing goal of keeping things named and dumpable
  (`strips()` / `named_values()` already do this for MDBE's columns).
- **Proven.** This is not a research toy — it's the literal architecture
  behind the world's best general-purpose compressors, which is a strong
  signal it captures real, useful structure in byte streams.
- **Naturally handles the grammar-mask idea from our earlier discussion.**
  Specialist C (the flag-based one) *is* a grammar mask — it just gets
  combined statistically instead of applied as a hard override.

## 5. Weaknesses / tradeoffs

- **Shallow per-specialist.** Each individual specialist only sees local or
  narrow structure. It won't "understand" a sentence's meaning the way a
  trained sequence model with a large hidden state can — it's good at
  statistical/local structure, weaker at long-range semantic structure.
- **More moving parts.** You now have several specialists *and* a mixer to
  maintain, instead of one model. Each specialist is simple, but there are
  more of them.
- **Bit-by-bit is 8 steps per byte.** Same tradeoff as before: predicting a
  byte one bit at a time means 8 small decisions instead of 1 — cheap here
  since each decision is tiny, but still 8x the *number* of steps.

## 6. Key terms to search if you want to go deeper

- **PAQ / cmix / zpaq** — the real-world compressors this idea comes from.
- **Context mixing** — the general name for "combine several predictors'
  probabilities with a learned mixer."
- **Logistic mixing / stretch-squash** — the specific math trick (step 3–4
  above) almost all of these tools use to combine probabilities.
- **Arithmetic coding** — the compression back-end that turns "probability
  of next bit" into an actual compressed file (not required for generation,
  only if you want compression, not just generation).
