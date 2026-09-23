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
