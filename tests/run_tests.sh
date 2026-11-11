#!/usr/bin/env bash
# Correctness test-suite.
#
# For every test file we compute the expected answer with an INDEPENDENT Python
# reference (tests/reference.py) and require that
#   1. the C++ serial program prints exactly the same thing, and
#   2. the MPI+OpenMP program prints exactly the same thing for many different
#      combinations of MPI ranks (P), OpenMP threads (T) and chunk sizes.
# The full word list is compared (--top 0), not only the first few words.
#
# Usage: bash tests/run_tests.sh        (run from the project root, after `make`)

set -u
cd "$(dirname "$0")/.."

SERIAL=build/wc_serial
PAR=build/wc_mpi
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Open MPI needs these flags when running more ranks than cores / as root (e.g. in Docker).
MPIRUN="mpirun"
if mpirun --version 2>&1 | grep -qi "open mpi"; then
    MPIRUN="mpirun --oversubscribe"
    [ "$(id -u)" = "0" ] && MPIRUN="$MPIRUN --allow-run-as-root"
fi

# ---------------------------------------------------------------- test inputs
D="$TMP/in"; mkdir -p "$D"
: > "$D/empty.txt"                                              # empty file
printf 'hello' > "$D/one_word_no_newline.txt"                   # single word, no trailing newline
printf 'a b c' > "$D/tiny_fewer_bytes_than_ranks.txt"           # file smaller than the number of ranks
printf '...,,, --- !!! \n\n\t  ' > "$D/only_separators.txt"     # no words at all
printf 'The the THE tHe\r\nCRLF line\r\nend' > "$D/case_and_crlf.txt"
printf "don't stop-me 3.14 foo_bar a1b2 x" > "$D/punctuation_digits.txt"
printf 'Привет мир привет\nПривет, Мир!\n' > "$D/utf8_cyrillic.txt"   # UTF-8 text
python3 - "$D" <<'PY'
import random, sys
d = sys.argv[1]
# one 200 KB word: spans several ranks AND is longer than the 64 KB tail block
open(f"{d}/very_long_word.txt", "w").write("a" * 200_000 + " b " + "a" * 70_000 + "\n")
# many one-letter words: every chunk boundary is next to a word
rng_letters = random.Random(1)
open(f"{d}/single_letters.txt", "w").write(" ".join(rng_letters.choice("abcde") for _ in range(20_000)))
# words of random length separated by random separators (stress for boundaries)
rng = random.Random(2)
seps = [" ", "  ", "\n", ",", ". ", "-", "\t"]
words = ["".join(rng.choice("abcXYZ019") for _ in range(rng.randint(1, 12))) for _ in range(30_000)]
open(f"{d}/random_boundaries.txt", "w").write("".join(w + rng.choice(seps) for w in words))
PY
python3 scripts/gen_corpus.py "$D/zipf_4mb.txt" --size-mb 4 --vocab 5000 >/dev/null

# ------------------------------------------------------------------- running
pass=0; fail=0
check() {  # check <label> <expected-file> <command...>
    local label="$1" expected="$2"; shift 2
    if "$@" > "$TMP/out.txt" 2>/dev/null && cmp -s "$expected" "$TMP/out.txt"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1)); echo "FAIL: $label"
    fi
}

# (ranks threads chunk-bytes) combinations; tiny chunks force many chunk boundaries
CONFIGS=("1 1 1048576" "2 1 1048576" "3 2 7" "4 1 1" "5 3 1000" "7 2 64" "2 4 13" "8 1 1048576")

for f in "$D"/*.txt; do
    name=$(basename "$f")
    python3 tests/reference.py --top 0 "$f" > "$TMP/expected.txt"
    check "serial  $name" "$TMP/expected.txt" $SERIAL --top 0 "$f"
    for cfg in "${CONFIGS[@]}"; do
        set -- $cfg
        check "mpi P=$1 T=$2 chunk=$3  $name" "$TMP/expected.txt" \
              env OMP_NUM_THREADS=$2 $MPIRUN -np $1 $PAR --threads $2 --chunk $3 --top 0 "$f"
    done
    # --top N must equal the first N lines of the full ordering
    python3 tests/reference.py --top 5 "$f" > "$TMP/expected_top5.txt"
    check "mpi top5 $name" "$TMP/expected_top5.txt" \
          env OMP_NUM_THREADS=2 $MPIRUN -np 3 $PAR --threads 2 --top 5 "$f"
done

echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ]
