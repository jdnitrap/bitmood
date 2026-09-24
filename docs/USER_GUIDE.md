# cmix-bit user guide

`cmix-bit` is a command-line tool that learns patterns in files one bit at a
time and then uses what it learned to **write new text** (or pictures, or
sounds), **help you write**, **compress files**, and **show you the
structure** hidden in a file's bytes.

It runs on a normal CPU, needs no internet, and learns only from the files
you give it.

The example outputs in this guide are real: they come from running the
commands on this repository's own `README.md` and `CONVERSATION.md`.

---

## Contents

1. [Getting started](#1-getting-started)
2. [Your first text generator](#2-your-first-text-generator)
3. [Getting good output](#3-getting-good-output)
4. [Creative writing](#4-creative-writing)
5. [Writing together with the model](#5-writing-together-with-the-model)
6. [Pictures and sounds](#6-pictures-and-sounds)
7. [Looking at files the ImHex way](#7-looking-at-files-the-imhex-way)
8. [The graph: words, sound shapes, pixel runs](#8-the-graph-words-sound-shapes-pixel-runs)
9. [Compressing files](#9-compressing-files)
10. [Troubleshooting](#10-troubleshooting)
11. [Command reference](#11-command-reference)

---

## 1. Getting started

### Build

You need `g++` (C++17) and `make`. From the repository folder:

```
make
make test
```

`make test` runs 84 checks and should end with `all tests passed`.
It needs `python3` for a few of them.

Run the program with no arguments to see every command:

```
./cmix-bit
```

### Words used in this guide

| Word | Meaning |
|---|---|
| **memory** | A file (you choose the name, e.g. `brain.bin`) that holds everything the model has learned. You create it with `train` and use it with `generate`, `write` and `info`. |
| **training** | Showing the model files so it learns their patterns. It keeps learning every time you train the same memory again. |
| **bits per byte** | How surprised the model was by the data. 8 = no idea at all; lower = it predicted well. |
| **prompt** | Text you give `generate` to start from. |
| **specialist** | One small part of the model that looks at the data in one particular way (the last few letters, the current word, the byte above in a grid, ...). A mixer combines their votes. |

---

## 2. Your first text generator

### Step 1: train a memory

Pick one or more text files. The more text, the better the output. Here we
use this repository's own documents:

```
./cmix-bit train --state brain.bin --table-bits 20 README.md CONVERSATION.md
```

```
new text memory brain.bin
  README.md                                     15007 bytes   2.882 bits/byte
  CONVERSATION.md                                6752 bytes   2.681 bits/byte
saved brain.bin: 21759 bytes learned in total, 2.820 bits/byte average
```

`--table-bits 20` makes a 12 MB memory file, plenty for a few hundred KB of
text. Leave it out for the default (about 36 MB, for several MB of text).
It can only be set when the memory is created.

### Step 2: train more whenever you like

Training an existing memory continues where it stopped:

```
./cmix-bit train --state brain.bin docs/README.md
```

```
loaded text memory brain.bin (21759 bytes learned so far)
  docs/README.md                                  198 bytes   1.710 bits/byte
saved brain.bin: 21957 bytes learned in total, 2.810 bits/byte average
```

Training in one run or in several runs gives exactly the same memory.

### Step 3: look at the memory

```
./cmix-bit info brain.bin
```

```
type           text
bytes learned  21957
bits/byte      2.810 average while learning, 2.552 recently
history        21957 bytes kept
tables         2^20 slots (8 MB), history up to 16 MB
specialists    O0 D1 A D3 D4 D5 D6 D8 W1 W2 C1 C2 E0u E0r E1u E1r E2u E2r B1 B2 bias
grid views     [line rows] [no width yet] [no width yet]
```

"Recently" is how well it predicts now that it has learned; it is usually
lower than the average, which includes the start when it knew nothing.

### Step 4: generate

```
./cmix-bit generate 200 "The model " --state brain.bin --seed 1
```

```
The model believes and the bytes and same specialist C xaa, B per byte.**  memoricepejs. trend-charset, learned, phaped of the probability to specialist CLSQ.
* specialist CLI is + mixer weights + hed compresso
```

- `200` is how many bytes to write after the prompt.
- `--seed` picks the random choices; the same seed gives the same text, a
  different seed gives different text.
- Generating never changes the memory.

With default settings the output wanders. The next section shows how to
make it better.

---

## 3. Getting good output

### The four settings to try first

```
./cmix-bit generate 300 "The model " --state brain.bin \
    --temp 0.6 --novelty 16 --best-of 6 --max-line 70 --seed 1
```

```
The model believerage in `).
- `
 **.

# CPU is coded payloats with a newis significant bit.

## Data the one complete next by the grid saves 16-bit compress only,
  compress only bytes the bit at at all and columns sittinverse of byt
  (neede the word in product) and new time bit into and with a learned
```

| Setting | What it does | Try |
|---|---|---|
| `--temp T` | How daring each choice is. Below 1 is safer and more repetitive; above 1 is wilder. | 0.5 to 0.8 |
| `--novelty N` | Never copy more than N bytes in a row from the training text, or from its own earlier output. Stops it reciting or looping. | 12 to 24 |
| `--best-of N` | Writes N versions of each line and keeps the best one (not too dull, no garbage, no long copies, normal line length). | 4 to 12 |
| `--max-line N` | Lines end by column N. | 60 to 90 |

### See what happened: `--stats`

Add `--stats` to see numbers on the error output:

```
line 1: best of 6, 1.70 bits/byte (target 2.55), 0 spikes, copy 10
...
generated 360 bytes | model surprise 1.425 bits/byte | longest copy from training text: 16 bytes
```

- **model surprise**: how unusual the output is to the model. Very low
  means dull and repetitive; much higher than the memory's "recently"
  value means nonsense.
- **spikes**: bytes the model found extremely unlikely (usually garbage).
- **longest copy**: the longest stretch copied word for word from the
  training text.

### If the output is ...

| Problem | Fix |
|---|---|
| repeating the same phrase | add `--novelty 16`, or raise `--temp` a little |
| copying the training text | lower `--novelty` (e.g. 12) |
| random letters and symbols | lower `--temp`; if you use `--novelty`, **raise** it (see below); train on more text |
| very long lines or no line breaks | `--max-line 70` |
| strange characters | the default `--charset seen` only uses characters from the training text; `--charset ascii` allows plain ASCII only |

**Small memories and `--novelty`:** if the memory is small or repetitive
(for example one short poem repeated), almost everything the model believes
*is* a copy, so a low `--novelty` forbids nearly everything and the output
turns to noise. Raise it (24 or more) or train on more text.

### Other sampling settings

- `--top-k K`: only choose among the K most likely next bytes.
- `--top-k 1`: always the most likely byte (no randomness).
- `--top-p P`: only choose among the most likely bytes that together make up P (e.g. 0.95) of the probability.

---

## 4. Creative writing

All of these combine with the settings above.

### Acrostic: lines start with the letters of a word

```
./cmix-bit generate 300 "" --state brain.bin --acrostic BITS --max-line 50 \
    --temp 0.6 --novelty 16 --seed 2
```

```
byte sampling frrostical the byte d bat: `state as
it saves and state brain.bin one stream start.
The file header? (default) | and learned bytd: (`-
several narrow widths |
...
```

Lines 1-4 start with B, I, T, S (either case). Use an empty prompt `""` so
the first line is the first letter.

### Other line and word rules

| Option | Effect |
|---|---|
| `--line-start "#-"` | every line starts with one of these characters |
| `--words list.txt` | only words from the list (one or more per line); between words only spaces and ordinary punctuation |
| `--rhyme` | rough rhyme in pairs of lines (AABB): the second line ends on a word ending like the first line's last word. Works best with `--best-of 8` or more. |

### Poems

Train a memory on poems (the more the better), then:

```
./cmix-bit train --state poems.bin --table-bits 20 mypoems.txt
./cmix-bit generate 400 "" --state poems.bin --temp 0.8 --novelty 24 \
    --max-line 60 --rhyme --best-of 12 --seed 11
```

With a small set of rhyming couplets, this produced lines such as
"and all the once against the nighte, / and all the birds had the birds
had growing late." The rhyme is rough (it compares the last two letters)
and the text keeps the style of the training poems, not their meaning.

### Blending two memories

Train two memories on different kinds of text and write from both:

```
./cmix-bit generate 400 "The " --state prose.bin --state code.bin --blend 0.7,0.3
```

- `--blend` gives one weight per `--state`, in the same order.
- `--blend-mode mix` (default): each memory keeps its confident choices,
  so the styles take turns.
- `--blend-mode product`: only what both memories agree on stays likely,
  a compromise style.

Blending very different styles gives noisier text; start with weights
like 0.8,0.2.

---

## 5. Writing together with the model

```
./cmix-bit write --state brain.bin --out draft.txt
```

Type normally. The model's guess for what comes next appears in grey after
your cursor.

| Key | Does |
|---|---|
| **Tab** | accept the whole grey suggestion |
| **→** (right arrow) | accept the next word of it |
| **Backspace** | delete within the word you are typing |
| **Enter** | new line |
| **Ctrl-S** | save the memory now |
| **Ctrl-D**, **Esc** or **Ctrl-C** | save and quit |

- Every word you finish (with a space, punctuation, Enter or an accepted
  suggestion) is **learned**: the memory picks up your writing style as you go.
- Backspace only works inside the word you are typing, because finished
  words are already learned. It beeps otherwise.
- `--out draft.txt` saves what you wrote. Without it the text is only
  learned, not kept.
- `--no-save` tries things without changing the memory.
- `--suggest N` sets how many characters the grey suggestion shows (default 24).
- If the memory doesn't exist yet, `write` creates a new one.

`write` needs a real terminal (it won't run with piped input).

---

## 6. Pictures and sounds

Tell the model what kind of data it gets with `--type`. It still learns
from bytes, but it knows where to look (the pixel above, the previous
sample), and it keeps each kind in its own memory.

### Pictures

`cmix-bit` reads **binary PPM** (colour, `P6`) and **PGM** (grey, `P5`),
8 bits per channel. To convert other formats you can install ImageMagick
(`sudo apt install imagemagick`) or use GIMP (File → Export As → `.ppm`):

```
convert photo.jpg -resize 128x photo.ppm
```

All pictures in one memory must have the same width and colour type. Small
pictures (around 64-256 pixels wide) train and generate quickly.

```
./cmix-bit train --state pics.bin --type image --table-bits 20 a.ppm b.ppm
./cmix-bit generate --state pics.bin --height 96 --out new.ppm --temp 0.7 --seed 3
convert new.ppm new.png          # to view it anywhere
```

The generated picture is new, not a copy: it has the training pictures'
colours, smooth areas and textures. Try `--temp` between 0.5 and 0.9.

Raw pixel files (no header) need the shape: `--width 128 --channels 3`.

### Sounds

`cmix-bit` reads **16-bit PCM WAV**, mono or stereo. To convert with ffmpeg
(`sudo apt install ffmpeg`):

```
ffmpeg -i song.mp3 -ac 1 -ar 16000 -sample_fmt s16 song.wav
```

```
./cmix-bit train --state sound.bin --type audio --table-bits 20 song.wav
./cmix-bit generate --state sound.bin --seconds 2 --out new.wav --temp 0.4 --seed 2
```

Honest expectation: generated sound is mostly noise shaped like the
training sound. Lower temperatures (0.3-0.5) keep more of the tone.

### Any other file: `raw`

`--type raw` works on anything (programs, databases, save files). It tries
every grid view, including 16-, 32- and 64-bit words.

---

## 7. Looking at files the ImHex way

A hex editor like ImHex shows a file as rows and columns. With the right
column count, repeating records line up as vertical stripes. `cmix-bit`
finds those row widths by itself, and `compare` shows what it found:

```
./cmix-bit compare README.md src/model/model.cpp
```

```
                       README.md                model.cpp
size (bytes)           15007                    13413
bits per byte          2.879                    2.130      <- lower = more predictable
  without grid views   2.879                    2.134
  grid saves           0.0%                     0.2%
view that won most     no grid                  no grid

  O0 order-0                   ███░░░░░  4.87    ███░░░░░  5.18
  A recent (order-2)           ████░░░░  3.95    █████░░░  3.27
  W word                       ████░░░░  4.15    ████░░░░  3.81
  E line rows (above)          ██░░░░░░  5.70    ██░░░░░░  5.68
  B long match (learned)       ██░░░░░░  6.30    ███░░░░░  4.65
  ...
  auto widths at end of file   none, none               17, 708
```

How to read it:

- **bits per byte**: lower means the file is more predictable.
- **grid saves**: how much the row-and-column views help. Prose gains
  nothing; tables, records and structured files gain a lot (7% on an HTML
  report with a 29-byte data record, 27% on a file of fixed-size
  records).
- **specialists alone**: how well each specialist would do on its own.
  A fuller bar is better. This shows *what kind* of pattern the file has:
  high orders and long match mean repetition, the word specialist means
  prose, the grid means records.
- **segments** (further down in the real output): regions of the file and
  which view won there, e.g. `0x0879-0x1073  width 24`, meaning the model
  saw 24-byte records in that region. You can open the file in ImHex, set
  24 columns, and see the stripes.
- **auto widths**: the row widths it found.
- **cross-file**: whether learning one file first helps with the other.

For binary files use `--type raw`. On a compiled program it finds the
24-byte symbol-table records and 64-bit pointer areas on its own.

---

## 8. The graph: words, sound shapes, pixel runs

The graph is an optional part of a memory that remembers *what usually
comes next* in bigger units than bytes:

| Type | A node is | An edge means |
|---|---|---|
| text | a word | this word was followed by that word |
| audio | the shape of a 20 ms slice (loudness, pitch, rising/steady/fading) | this sound shape was followed by that one |
| image | the shape of 8 pixels in a row (brightness, slope, texture, colour) | with this run above and that run to the left, this run came |

An edge only counts once it has been seen twice ("needs a yes before it
is a fact"); `--graph-confirm N` changes that number.

### Make a memory with a graph

`--graph` only works when a memory is created:

```
./cmix-bit train --state brain.bin --table-bits 20 --graph README.md CONVERSATION.md docs/USER_GUIDE.md
./cmix-bit info brain.bin
```

```
graph          5773 nodes, 10113 edges (1248 confirmed, needs 2 sightings), 1205 words
```

### Look inside

```
./cmix-bit graph brain.bin
```

```
word graph: 5773 nodes, 10113 edges, 1248 confirmed (an edge needs 2 sightings)

most frequent words (1205 different):
     306  the
     158  a
     111  bit
...
strongest links:
      54  cmix -> bit
      29  the -> model
      21  brain -> bin
...
```

One word's neighbours:

```
./cmix-bit graph brain.bin memory
```

```
'memory' seen 53 times

after 'memory':
       5  file
       4  is
       2  type
...
before 'memory':
      16  the
       3  text
       3  state
```

For audio and image memories, `graph` describes the shapes instead, e.g.
`[loud 6/7, ~565 Hz, steady] -> [loud 5/7, ~565 Hz, fading]`.

### Plan while generating

`--graph-plan` makes the generator pick, at the start of each word (slice,
run), what should come next from the graph, and lean toward it:

```
./cmix-bit generate 300 "The " --state brain.bin --temp 0.7 --novelty 16 --graph-plan
```

| Option | Effect |
|---|---|
| `--graph-plan` | plan with the default strength for the memory's type |
| `--plan-strength S` | how hard to lean toward the plan (defaults: text 4, audio 100, image 2) |
| `--plan-temp T` | how adventurous the plans are (default: text follows `--temp`; audio and image 1 = as often as in the training data) |

What to expect, from measurements:

- **Text:** word-to-word flow improves. Over 8 samples, the share of word
  pairs that also occur in the training text went from 52% to 65% (strength
  4) and 72% (strength 20). It still doesn't give meaning.
- **Audio:** it gets generated sound out of long silences and into a
  note / pause rhythm (2-3 times more loud slices), but the notes are noisy.
- **Images:** planning makes pictures flatter and greyer, and strong
  planning adds horizontal bands. The graph itself (`graph`) is still
  interesting to look at; for generating pictures, leave planning off.
- **Compression:** the graph changes compressed size by less than 0.1%.

### The spiking network (`--snn`)

`--snn` (together with `--graph`, when creating a memory) turns the graph
into a small spiking brain:

1. When a word (sound slice, pixel run) finishes, its neuron **spikes**.
2. The spike **charges** the neurons its edges point to (the likely next ones).
3. Charge **leaks** each step (`--snn-leak`, default 0.6), so words from a
   few steps back still count; the plain graph only looks at the last two.
4. The most-charged neurons are the prediction: vote **N**, and the source
   of plans with `--graph-plan`.

It learns from the model's own surprise: a word predicted better than usual
strengthens the connections that predicted it, a word predicted worse
weakens them, and a surprise also strengthens the path to what really came.

```
./cmix-bit train --state brain.bin --table-bits 20 --graph --snn mytext.txt
./cmix-bit generate 300 "The " --state brain.bin --temp 0.7 --novelty 16 --graph-plan
```

**Finding where data changes.** A sudden rise in surprise marks a "new
region". `compare` lists these points:

```
./cmix-bit compare --graph --snn joined.txt other.txt
```

```
  surprise jumps (SNN change detection): 7 at 0x0000a3, 0x010f8c, 0x029d13, 0x02a9a0, 0x03ea33, 0x042fee, 0x048970
```

That was four files joined together (licenses, Python, HTML, markdown):
`0x010f8c` and `0x042fee` are the joins (within 54 and 176 bytes),
`0x029d13` is a join between two Python files inside one file, and
`0x048970` is where the HTML's CSS turns into JavaScript. It reacts to
*more surprise*, not to every change of kind: text turning into plain
numbers isn't flagged, text turning into random-looking data is.

What the SNN does, measured against the plain graph:

| | plain graph | graph + SNN |
|---|---|---|
| vote alone (licenses) | 6.83 bits/byte | 5.81 |
| overall compression | | 0.1-1% smaller |
| planned text: real words | 92.5% | 93.2% |
| planned text: known word pairs | 64.7% | 55.4% |
| generated audio: loud slices (temp 1, training 78%) | 47% | 61% |

## 9. Compressing files

```
./cmix-bit compress README.md readme.cmxb
./cmix-bit decompress readme.cmxb readme.txt
```

```
in  15007 bytes
out 5430 bytes (header+payload)
wrote 15007 bytes
```

The file comes back identical. Compression is strong (better than `xz -9e`
on text and binaries) but slow (roughly 150 KB per second), so it suits
small and medium files.

- Use `--type image`, `--type audio` or `--type raw` for those kinds of files.
  The type is stored in the compressed file, so `decompress` needs no options.
- `--lstm 32` adds the neural-network specialist: about 1% smaller, about 8x slower.
- Decompress with the same build of `cmix-bit` that compressed. Files from
  older versions of this tool can't be read (it says so).

---

## 10. Troubleshooting

| Message | Meaning / fix |
|---|---|
| `cannot open brain.bin` | The memory doesn't exist. Create it with `train` first (or check the path). |
| `checksum mismatch (file damaged)` | The memory file is damaged (e.g. copied halfway). Retrain it. |
| `state file version N, this build reads M; retrain with this build` | The memory was made by a different version of `cmix-bit`. Retrain. |
| `brain.bin is a text memory, not image` | Each memory has one type; use a separate memory for each kind of data. |
| `... is 64 px wide ...; this memory is for 128 px` | All pictures in one memory need the same width and colour type. |
| `--blend needs one weight per --state` | Give exactly one weight for each `--state`. |
| `image output needs --out FILE` | Pictures and sounds are written to a file, not the screen. |
| `write needs an interactive terminal` | Run `write` in a terminal, not with piped input. |
| `... has no graph (train a new memory with --graph)` | `--graph-plan` and `graph` need a memory created with `--graph`. |
| `... already exists without a graph` | `--graph` can't be added to an existing memory; train a new one. |
| `--snn needs --graph` / `... already exists without an SNN` | `--snn` only works together with `--graph`, on a new memory. |
| `the constraints leave no byte that may come next` | Your rules contradict each other (e.g. a word list with no word starting with the acrostic letter). Loosen one. |

**Speed.** Training runs at about 150 KB per second (about 20 KB per second
with `--lstm`). Generating is fast for text; `--best-of N` takes N times
longer.

**Memory size.** A memory file is about 36 MB by default. With
`--table-bits 18` it is about 6 MB, with `20` about 12 MB. More bits help
only when you train on several MB of text.

**Output quality.** The model learns spelling, words, phrases and style, not
meaning. It won't keep a story or an argument going beyond about a
sentence. More training text, of one consistent style, helps most.

---

## 11. Command reference

### `train`
```
cmix-bit train --state MEMORY [--type text|image|audio|raw] [--table-bits 16..28]
               [--lstm N] [--graph [--graph-confirm N] [--snn [--snn-leak L]]]
               [--width W --channels 1|3] FILE...
```
Creates the memory if it doesn't exist, otherwise keeps learning.
`--type`, `--table-bits`, `--lstm`, `--graph`, `--snn` and the shape only apply when creating.

### `generate`
```
cmix-bit generate NBYTES [PROMPT] --state MEMORY [options]
cmix-bit generate --state IMAGE_MEMORY --height H --out new.ppm [options]
cmix-bit generate --state AUDIO_MEMORY --seconds S --out new.wav [options]
```
| Option | Default | |
|---|---|---|
| `--state FILE` (repeatable) | none (learns from the prompt only) | memory to write from |
| `--blend w1,w2,..` | all 1 | weight per memory |
| `--blend-mode mix\|product` | mix | how memories combine |
| `--temp T` | 1 | daring |
| `--top-k K`, `--top-p P` | off | limit choices to the most likely |
| `--seed N` | fixed | random choices |
| `--charset seen\|utf8\|ascii\|any` | seen (text; utf8 while the memory is empty), any (other) | allowed characters |
| `--novelty N` | off | longest allowed copy |
| `--line-start CHARS` / `--acrostic WORD` | off | how lines start |
| `--max-line N` | off | line length |
| `--words FILE` | off | allowed words |
| `--rhyme` | off | rough AABB rhyme |
| `--best-of N` | 1 | candidates per line |
| `--graph-plan` | off | plan from the graph (memory made with `--graph`) |
| `--plan-strength S`, `--plan-temp T` | per type | how hard / how adventurously to plan (text: `--temp`; audio, image: 1) |
| `--out FILE` | screen | write to a file |
| `--stats` | off | print statistics |

If you give only `NBYTES` and no prompt, the prompt is "The quick brown fox ".

### `write`
```
cmix-bit write --state MEMORY [--out FILE] [--suggest N] [--charset ...] [--no-save]
```

### `info`
```
cmix-bit info MEMORY
```

### `graph`
```
cmix-bit graph MEMORY [WORD] [--top N]
```

### `compare`
```
cmix-bit compare [--type text|image|audio|raw] [--width W --channels C] [--lstm N] [--graph [--snn]] FILE_A FILE_B
```

### `compress` / `decompress`
```
cmix-bit compress [--type text|image|audio|raw] [--width W --channels C] [--lstm N] [--graph [--snn]] IN OUT.cmxb
cmix-bit decompress IN.cmxb OUT
```

### `demo`
```
cmix-bit demo ["some text"]
```
Prints every bit's prediction from every specialist, to see the model at work.
