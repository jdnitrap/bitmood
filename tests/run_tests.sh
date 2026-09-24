#!/usr/bin/env bash
# End-to-end tests for cmix-bit. Run from the repo root: `make test`.
# Each check prints PASS/FAIL; the script exits non-zero if anything failed.
set -u
BIN=./cmix-bit
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }

# --- test data -------------------------------------------------------------
cat README.md CONVERSATION.md > "$T/text.txt"
for i in $(seq 40); do printf 'The quick brown fox jumps over the lazy dog. '; done > "$T/rep.txt"
head -c 4000 /dev/urandom > "$T/rand.bin"
: > "$T/empty.bin"
head -c 3000 /dev/zero > "$T/zero.bin"
printf 'x' > "$T/one.bin"

echo "compress / decompress round trip"
for f in text.txt rep.txt rand.bin empty.bin zero.bin one.bin; do
  $BIN compress "$T/$f" "$T/$f.cmxb" 2>/dev/null &&
    $BIN decompress "$T/$f.cmxb" "$T/$f.out" 2>/dev/null
  check "$f round trip is byte-identical" "cmp -s '$T/$f' '$T/$f.out'"
done
size_rep=$(stat -c%s "$T/rep.txt.cmxb")
check "repeated text compresses (got $size_rep bytes from 1800)" "[ $size_rep -lt 200 ]"
check "decompress rejects a non-CMXB file" "! $BIN decompress '$T/rand.bin' '$T/x' 2>/dev/null"

echo "saved memory"
$BIN train --state "$T/one.st" README.md CONVERSATION.md 2>/dev/null
$BIN train --state "$T/two.st" README.md 2>/dev/null
$BIN train --state "$T/two.st" CONVERSATION.md 2>/dev/null
check "training in one run == training in two runs" "cmp -s '$T/one.st' '$T/two.st'"
learned=$(( $(stat -c%s README.md) + $(stat -c%s CONVERSATION.md) ))
check "info reads the memory ($learned bytes)" "$BIN info '$T/one.st' | grep -q 'bytes learned  $learned\$'"
head -c 1000 "$T/one.st" > "$T/trunc.st"
check "truncated memory is rejected" "! $BIN info '$T/trunc.st' 2>/dev/null"
cp "$T/one.st" "$T/flip.st"
printf '\x55' | dd of="$T/flip.st" bs=1 seek=5000 conv=notrunc 2>/dev/null
check "damaged memory is rejected (checksum)" "! $BIN info '$T/flip.st' 2>/dev/null"
check "unknown --type is rejected" "! $BIN train --state '$T/bad.st' --type banana README.md 2>/dev/null"

echo "generate"
g1=$($BIN generate 200 "The " --state "$T/one.st" --seed 7 | md5sum)
g2=$($BIN generate 200 "The " --state "$T/one.st" --seed 7 | md5sum)
g3=$($BIN generate 200 "The " --state "$T/one.st" --seed 8 | md5sum)
check "same --seed gives the same text" "[ '$g1' = '$g2' ]"
check "different --seed gives different text" "[ '$g1' != '$g3' ]"
check "generate does not modify the memory" "cmp -s '$T/one.st' '$T/two.st'"
check "--temp 0 is rejected" "! $BIN generate 10 --temp 0 >/dev/null 2>&1"

echo "generation controls"
$BIN generate 400 "The " --state "$T/one.st" --temp 0.9 --seed 5 > "$T/g.txt"
check "default output is valid UTF-8" "python3 -c 'open(\"$T/g.txt\",\"rb\").read().decode(\"utf-8\")'"
check "default output only uses bytes seen in training" "python3 - '$T/g.txt' README.md CONVERSATION.md <<'PY'
import sys
out = open(sys.argv[1], 'rb').read()
seen = set(open(sys.argv[2], 'rb').read() + open(sys.argv[3], 'rb').read() + b'The \n')
sys.exit(0 if set(out) <= seen else 1)
PY"
$BIN generate 400 "The " --state "$T/one.st" --charset ascii --temp 1.2 --seed 5 > "$T/a.txt"
check "--charset ascii gives printable ASCII only" "! LC_ALL=C grep -q '[^[:print:][:space:]]' '$T/a.txt'"
$BIN generate 600 "The " --state "$T/one.st" --temp 0.5 --novelty 12 --seed 9 --stats > "$T/n.txt" 2> "$T/n.err"
copy=$(grep -o 'longest copy from training text: [0-9]*' "$T/n.err" | grep -o '[0-9]*$')
check "--novelty 12 caps copies from training text (longest $copy)" "[ '$copy' -le 12 ]"
check "--novelty 12 blocks repeating its own 13-byte runs" "python3 - '$T/n.txt' <<'PY'
import sys
s = open(sys.argv[1], 'rb').read()[4:]   # skip the prompt
runs = [s[i:i+13] for i in range(len(s) - 12)]
sys.exit(0 if len(runs) == len(set(runs)) else 1)
PY"
k1=$($BIN generate 100 "The " --state "$T/one.st" --top-k 1 --seed 1 | md5sum)
k2=$($BIN generate 100 "The " --state "$T/one.st" --top-k 1 --seed 2 | md5sum)
check "--top-k 1 is greedy (seed does not matter)" "[ '$k1' = '$k2' ]"

echo "creativity controls"
$BIN train --state "$T/code.st" Makefile tests/run_tests.sh 2>/dev/null
s1=$($BIN generate 200 "The " --state "$T/one.st" --charset any --seed 3 | md5sum)
s2=$($BIN generate 200 "The " --state "$T/one.st" --state "$T/code.st" --blend 1,0 --charset any --seed 3 | md5sum)
check "--blend 1,0 equals the first memory alone" "[ '$s1' = '$s2' ]"
s3=$($BIN generate 200 "The " --state "$T/one.st" --state "$T/code.st" --blend 0.5,0.5 --charset any --seed 3 | md5sum)
check "--blend 0.5,0.5 differs from one memory" "[ '$s1' != '$s3' ]"
check "--blend-mode product runs" "$BIN generate 50 'The ' --state '$T/one.st' --state '$T/code.st' --blend-mode product >/dev/null"
check "blending text with a missing weight is rejected" "! $BIN generate 5 --state '$T/one.st' --state '$T/code.st' --blend 1 2>/dev/null"
$BIN generate 400 "" --state "$T/one.st" --acrostic MIXER --max-line 50 --seed 2 > "$T/acro.txt"
check "--acrostic MIXER starts lines M,I,X,E,R" "[ \"\$(head -5 '$T/acro.txt' | cut -c1 | tr -d '\n' | tr a-z A-Z)\" = MIXER ]"
check "--max-line 50 keeps every line within 50 bytes" "! awk 'length(\$0) > 50 {bad=1} END {exit !bad}' '$T/acro.txt'"
$BIN generate 300 "" --state "$T/one.st" --line-start "#-" --max-line 40 --seed 2 > "$T/ls.txt"
check "--line-start '#-' starts every line with # or -" "! grep -v '^[#-]' <(grep -v '^\$' '$T/ls.txt') | grep -q ."
printf 'the model learns bits bytes and a mixer of the text\n' > "$T/words.txt"
$BIN generate 300 "the " --state "$T/one.st" --words "$T/words.txt" --seed 2 > "$T/w.txt"
check "--words: every word comes from the list" "python3 - '$T/w.txt' '$T/words.txt' <<'PY'
import re, sys
allowed = set(open(sys.argv[2]).read().split())
words = re.findall(r\"[A-Za-z']+\", open(sys.argv[1]).read())
sys.exit(0 if words and all(w.lower() in allowed for w in words) else 1)
PY"
b1=$($BIN generate 200 "The " --state "$T/one.st" --best-of 4 --seed 6 | md5sum)
b2=$($BIN generate 200 "The " --state "$T/one.st" --best-of 4 --seed 6 | md5sum)
check "--best-of 4 is repeatable with the same seed" "[ '$b1' = '$b2' ]"
check "generate still leaves the memory untouched" "cmp -s '$T/one.st' '$T/two.st'"

echo "interactive write"
cp "$T/one.st" "$T/w.st"
check "write: typing, Tab, backspace, arrow, Enter, save" "timeout 60 python3 tests/write_test.py $BIN '$T/w.st' '$T/w_out.txt' >/dev/null"
check "generate works with no memory and an empty prompt" "$BIN generate 20 '' >/dev/null"
check "write refuses to run without a terminal" "! $BIN write --state '$T/w.st' </dev/null 2>/dev/null"

echo "data types: image, audio, raw"
python3 - "$T" <<'PY'
import math, random, struct, sys
d = sys.argv[1]; random.seed(3)
def ppm(name, w, h, k):
    px = bytearray()
    for y in range(h):
        for x in range(w):
            px += bytes(max(0, min(255, int(v) + random.randint(-2, 2))) for v in
                        (120 + 90 * math.sin(x / 9 + k), 60 + y, 40 + x))
    open(f'{d}/{name}', 'wb').write(b'P6\n# test\n%d %d\n255\n' % (w, h) + px)
ppm('a.ppm', 64, 48, 1); ppm('b.ppm', 64, 48, 2); ppm('narrow.ppm', 32, 48, 3)
s = [int(8000 * math.sin(2 * math.pi * 220 * i / 8000) + random.gauss(0, 100)) for i in range(8000)]
data = struct.pack('<%dh' % len(s), *s)
open(f'{d}/tone.wav', 'wb').write(b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' +
    struct.pack('<IHHIIHH', 16, 1, 1, 8000, 16000, 2, 16) + b'data' + struct.pack('<I', len(data)) + data)
PY
for spec in "image a.ppm" "audio tone.wav" "raw a.ppm"; do
  set -- $spec
  $BIN compress --type $1 "$T/$2" "$T/typed.cmxb" 2>/dev/null && $BIN decompress "$T/typed.cmxb" "$T/typed.out" 2>/dev/null
  check "--type $1 $2 round trip is byte-identical" "cmp -s '$T/$2' '$T/typed.out'"
done
img=$($BIN compress --type image "$T/a.ppm" "$T/i.cmxb" 2>&1 | awk '/^out/{print $2}')
raw=$($BIN compress --type raw "$T/a.ppm" "$T/r.cmxb" 2>&1 | awk '/^out/{print $2}')
check "telling it 'image' beats 'raw' on a picture ($img < $raw)" "[ $img -lt $raw ]"
snd=$($BIN compress --type audio "$T/tone.wav" "$T/s.cmxb" 2>&1 | awk '/^out/{print $2}')
raw=$($BIN compress --type raw "$T/tone.wav" "$T/r.cmxb" 2>&1 | awk '/^out/{print $2}')
check "telling it 'audio' beats 'raw' on a sound ($snd < $raw)" "[ $snd -lt $raw ]"
$BIN train --state "$T/img.st" --type image --table-bits 18 "$T/a.ppm" "$T/b.ppm" 2>/dev/null
check "info shows the image shape" "$BIN info '$T/img.st' | grep -q '64 px wide, 3 channels'"
check "an image of another width is rejected" "! $BIN train --state '$T/img.st' '$T/narrow.ppm' 2>/dev/null"
$BIN generate --state "$T/img.st" --height 10 --out "$T/g.ppm" --seed 1
check "image generation writes a 64x10 PPM" "[ \"\$(head -c 13 '$T/g.ppm' | tr '\n' ' ')\" = 'P6 64 10 255 ' ] && [ \$(stat -c%s '$T/g.ppm') -eq \$((13 + 64 * 10 * 3)) ]"
$BIN train --state "$T/snd.st" --type audio --table-bits 18 "$T/tone.wav" 2>/dev/null
$BIN generate --state "$T/snd.st" --seconds 0.25 --out "$T/g.wav" --seed 1
check "audio generation writes a 0.25 s WAV" "[ \$(stat -c%s '$T/g.wav') -eq \$((44 + 2000 * 2)) ] && [ \"\$(head -c 4 '$T/g.wav')\" = RIFF ]"
check "image output without --out is rejected" "! $BIN generate --state '$T/img.st' --height 2 >/dev/null 2>&1"
check "a WAV is rejected as an image" "! $BIN compress --type image '$T/tone.wav' '$T/x' 2>/dev/null"

echo "level 2: LSTM"
$BIN compress --lstm 8 CONVERSATION.md "$T/l.cmxb" 2>/dev/null && $BIN decompress "$T/l.cmxb" "$T/l.out" 2>/dev/null
check "--lstm 8 round trip is byte-identical" "cmp -s CONVERSATION.md '$T/l.out'"
$BIN train --state "$T/l1.st" --lstm 8 --table-bits 18 README.md CONVERSATION.md 2>/dev/null
$BIN train --state "$T/l2.st" --lstm 8 --table-bits 18 README.md 2>/dev/null
$BIN train --state "$T/l2.st" CONVERSATION.md 2>/dev/null
check "LSTM memory: one run == two runs" "cmp -s '$T/l1.st' '$T/l2.st'"
check "info shows the LSTM" "$BIN info '$T/l1.st' | grep -q 'LSTM           8 cells'"
check "a different --lstm on an existing memory is rejected" "! $BIN train --state '$T/l1.st' --lstm 16 README.md 2>/dev/null"
check "--lstm 65 is rejected" "! $BIN compress --lstm 65 README.md '$T/x' 2>/dev/null"

echo "graph specialist: text"
$BIN train --state "$T/g1.st" --graph --table-bits 18 README.md CONVERSATION.md 2>/dev/null
$BIN train --state "$T/g2.st" --graph --table-bits 18 README.md 2>/dev/null
$BIN train --state "$T/g2.st" CONVERSATION.md 2>/dev/null
check "graph memory: one run == two runs" "cmp -s '$T/g1.st' '$T/g2.st'"
check "info shows the graph" "$BIN info '$T/g1.st' | grep -q '^graph .* confirmed'"
check "graph summary lists 'the' among frequent words" "$BIN graph '$T/g1.st' | grep -A3 'most frequent' | grep -q ' the$'"
check "graph WORD shows what follows it" "$BIN graph '$T/g1.st' the | grep -q '^after'"
check "graph rejects an unseen word" "! $BIN graph '$T/g1.st' zzqqxx 2>/dev/null"
$BIN train --state "$T/g3.st" --graph --graph-confirm 1000 --table-bits 18 README.md 2>/dev/null
check "--graph-confirm 1000 leaves every edge a candidate" "$BIN info '$T/g3.st' | grep -q '(0 confirmed'"
check "--graph-plan needs a graph" "! $BIN generate 10 'The ' --state '$T/one.st' --graph-plan 2>/dev/null"
p1=$($BIN generate 200 "The " --state "$T/g1.st" --graph-plan --seed 4 | md5sum)
p2=$($BIN generate 200 "The " --state "$T/g1.st" --graph-plan --seed 4 | md5sum)
check "--graph-plan is repeatable with the same seed" "[ '$p1' = '$p2' ]"
check "generate --graph-plan leaves the memory untouched" "$BIN train --state '$T/g4.st' --graph --table-bits 18 README.md CONVERSATION.md 2>/dev/null && cmp -s '$T/g1.st' '$T/g4.st'"
$BIN compress --graph CONVERSATION.md "$T/gc.cmxb" 2>/dev/null && $BIN decompress "$T/gc.cmxb" "$T/gc.out" 2>/dev/null
check "--graph round trip is byte-identical" "cmp -s CONVERSATION.md '$T/gc.out'"
check "--graph on a raw memory is rejected" "! $BIN train --state '$T/gr.st' --type raw --graph README.md 2>/dev/null"

echo "graph specialist: audio and image"
$BIN train --state "$T/ga1.st" --type audio --graph --table-bits 18 "$T/tone.wav" "$T/tone.wav" 2>/dev/null
$BIN train --state "$T/ga2.st" --type audio --graph --table-bits 18 "$T/tone.wav" 2>/dev/null
$BIN train --state "$T/ga2.st" "$T/tone.wav" 2>/dev/null
check "audio graph memory: one run == two runs" "cmp -s '$T/ga1.st' '$T/ga2.st'"
check "graph shows sound shapes" "$BIN graph '$T/ga1.st' | grep -q 'sound-shape graph' && $BIN graph '$T/ga1.st' | grep -q 'Hz'"
$BIN generate --state "$T/ga1.st" --seconds 0.25 --out "$T/ga.wav" --graph-plan --seed 1
check "audio --graph-plan writes a 0.25 s WAV" "[ \$(stat -c%s '$T/ga.wav') -eq \$((44 + 2000 * 2)) ]"
$BIN train --state "$T/gi1.st" --type image --graph --table-bits 18 "$T/a.ppm" "$T/b.ppm" 2>/dev/null
$BIN train --state "$T/gi2.st" --type image --graph --table-bits 18 "$T/a.ppm" 2>/dev/null
$BIN train --state "$T/gi2.st" "$T/b.ppm" 2>/dev/null
check "image graph memory: one run == two runs" "cmp -s '$T/gi1.st' '$T/gi2.st'"
check "graph shows pixel-run shapes" "$BIN graph '$T/gi1.st' | grep -q 'pixel-run graph' && $BIN graph '$T/gi1.st' | grep -q 'brightness'"
check "looking up a word in an image graph is rejected" "! $BIN graph '$T/gi1.st' the 2>/dev/null"
$BIN generate --state "$T/gi1.st" --height 8 --out "$T/gi.ppm" --graph-plan --seed 1
check "image --graph-plan writes a 64x8 PPM" "[ \$(stat -c%s '$T/gi.ppm') -eq \$((12 + 64 * 8 * 3)) ]"
for spec in "image a.ppm" "audio tone.wav"; do
  set -- $spec
  $BIN compress --type $1 --graph "$T/$2" "$T/gt.cmxb" 2>/dev/null && $BIN decompress "$T/gt.cmxb" "$T/gt.out" 2>/dev/null
  check "--type $1 --graph round trip is byte-identical" "cmp -s '$T/$2' '$T/gt.out'"
done

echo "grid views"
words=(alpha beta gamma delta)
for i in $(seq 400); do  # 23-byte records; the word and number vary without a longer period
  printf 'ID%05d|%-9s|%4d;' "$i" "${words[$(( (i * i * 31 + 7) % 97 % 4 ))]}" "$(( (i * i * 7919 + 13) % 10007 % 10000 ))"
done > "$T/rec23.txt"
$BIN compare "$T/rec23.txt" README.md > "$T/cmp.txt"
check "compare finds the 23-byte record width" "grep -q 'view that won most *width 23' '$T/cmp.txt'"
saves=$(grep 'grid saves' "$T/cmp.txt" | awk '{print $3}' | tr -d '%')
check "grid views help on records (saves $saves%)" "awk 'BEGIN{exit !($saves > 10)}'"
$BIN compress "$T/rec23.txt" "$T/rec.cmxb" 2>/dev/null && $BIN decompress "$T/rec.cmxb" "$T/rec.out" 2>/dev/null
check "records round trip with grid on" "cmp -s '$T/rec23.txt' '$T/rec.out'"

echo
if [ $fails -eq 0 ]; then echo "all tests passed"; else echo "$fails test(s) FAILED"; exit 1; fi
