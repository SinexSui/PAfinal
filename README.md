# Parallel Word Count (MPI + OpenMP)

Final project, *Parallelization of Algorithms* - group **MCS 2502**
Authors: **Khalelov Adilkhan**, **Bazarbayev Temirlan**

A C++ program that counts the words of a (large) text file and reports the total number
of words, the number of distinct words and the **top-N most frequent words**. It exists in two versions:

| Program | What it is |
|---|---|
| `build/wc_serial` | sequential baseline and reference for speedup |
| `build/wc_mpi` | hybrid parallel version: **MPI** between processes + **OpenMP** between threads |

## 1. Requirements

* C++17 compiler with OpenMP (`g++`), an MPI implementation (Open MPI or MPICH) with `mpicxx`/`mpirun`
* Python 3 with `numpy` and `matplotlib` (only for tests, data generation and graphs)

**Ubuntu / Debian / WSL2 (recommended on Windows):**
```bash
sudo apt update
sudo apt install build-essential openmpi-bin libopenmpi-dev python3 python3-numpy python3-matplotlib
```
**macOS:** `brew install gcc open-mpi`, then build with `make OMPI_CXX=g++-14` (Apple clang has no OpenMP; use the Homebrew gcc version number you have).

## 2. Build, test, run

```bash
make                 # builds build/wc_serial and build/wc_mpi
make test            # correctness test-suite (see section 5)

# sequential
./build/wc_serial --top 10 data/sample.txt

# parallel: 4 MPI processes, each with 2 OpenMP threads  (8 cores in total)
mpirun -np 4 ./build/wc_mpi --threads 2 --top 10 data/sample.txt
```
Options (both programs print the same format, so outputs can be compared with `diff`):

| Option | Meaning | Default |
|---|---|---|
| `--top N` | print the N most frequent words (`0` = all words) | 20 |
| `--threads T` | OpenMP threads per MPI process (`wc_mpi` only) | `OMP_NUM_THREADS` |
| `--chunk BYTES` | size of the unit of work handed to a thread (`wc_mpi` only) | 1 MiB |

Output: `TOTAL_WORDS`, `UNIQUE_WORDS`, then one `word<TAB>count` line per word (highest count first,
ties alphabetical). Timing information goes to **stderr** as one `TIMING ...` line.

> Running Open MPI with more processes than cores: add `--oversubscribe`.
> Running hybrid jobs with Open MPI: add `--bind-to none`, otherwise Open MPI pins all threads of a process to one core.

## 3. How it works

### What counts as a word
A maximal run of ASCII letters/digits and non-ASCII bytes (so UTF-8 text like Cyrillic stays in one piece).
Everything else is a separator. ASCII letters are lower-cased (`The` = `the`).
`don't` is two words (`don`, `t`). Non-ASCII letters are not case-folded (documented limitation).

### Sequential algorithm
Read the file, scan it once, increment a counter in a hash map (`unordered_map<string, uint64>`), sort the entries.
Complexity: O(n) to count, O(u log u) to sort `u` distinct words.

### Parallel algorithm (4 phases)

```
file:   |----- rank 0 -----|----- rank 1 -----|----- rank 2 -----|----- rank 3 -----|
                           ^ a word may be cut here

Phase 1  READ      each rank reads ITS byte range with MPI-IO (nobody reads the whole file)
Phase 2  COUNT     inside a rank: OpenMP threads take 1 MiB chunks dynamically,
                   each thread has a PRIVATE hash map (no locks) -> merged into one map per rank
Phase 3  EXCHANGE  EXPLICIT OWNERSHIP: owner(word) = hash(word) % P
                   MPI_Alltoallv sends every (word, count) to its owner, the owner adds the counts
Phase 4  TOP-N     each rank keeps its N best words; MPI_Gatherv to rank 0, which prints the final top-N
                   (the global top-N is always inside the union of the local top-Ns because every word has ONE owner)
```

**Data ownership (explicit, as required):**
* *Input bytes*: rank `r` owns bytes `[start_r, end_r)`; inside a rank, every chunk is owned by exactly one thread at a time.
* *Words at borders*: **a word belongs to the chunk that contains its first byte.** A chunk skips the word it starts
  in the middle of, and reads a little past its end to finish its last word. The same rule (`scan_words()` in
  `src/common.hpp`) is used between ranks and between threads, so no word is lost or counted twice.
* *Vocabulary*: after the exchange, each distinct word lives on exactly one rank.

**Why this design:**
* Thread-private maps remove all synchronisation from the hot loop.
* Dynamic scheduling of chunks balances the load even if some parts of the text are denser than others.
* Hash-partitioned reduction (instead of sending everything to rank 0) keeps memory and network traffic spread over all ranks.
* Only MPI_THREAD_FUNNELED is needed: MPI is never called inside an OpenMP region.

**Known limitations / future work:** the merge of the thread maps inside one rank is sequential (cheap for natural
text whose vocabulary is small, but it would limit scaling for inputs with a huge number of distinct words);
whole-file text per rank is kept in memory; no non-ASCII case folding.

## 4. Benchmarks and graphs

```bash
python3 scripts/benchmark.py --mode strong --size-mb 512 --max-cores 8   # fixed input, more cores
python3 scripts/benchmark.py --mode weak   --size-mb 64  --max-cores 8   # input grows with cores
python3 scripts/plot.py                                                  # -> results/*.png + results/summary.md
```
Set `--max-cores` to the largest core count you want to test. For each core count the script measures pure MPI,
pure OpenMP and a hybrid configuration (3 repetitions + 1 warm-up, median is used). Input is a synthetic Zipf-distributed
text (`scripts/gen_corpus.py`); you can also pass real text with `--file` (e.g. concatenated books from Project Gutenberg).
Reported: speedup `T_serial / T(c)`, efficiency `speedup / c`, weak-scaling efficiency, and a time breakdown
(read / count / exchange). Timings are the slowest rank's time; files are read from the OS cache (warm-up run).

## 5. Correctness tests (`make test`)

`tests/run_tests.sh` compares the **complete word list** produced by the C++ programs with an independent Python
reference (`tests/reference.py`, uses `re` + `Counter`, shares no code with the C++ side) for 11 files:
empty file, one word without newline, file smaller than the number of ranks, only separators, mixed case + CRLF,
punctuation and digits, UTF-8 Cyrillic, a 200 KB word spanning several ranks (longer than the 64 KB tail block), 20 000 one-letter words,
random words with random separators, and a 4 MB Zipf text. Each file is run with 8 combinations of
(ranks, threads, chunk size) - including 1-byte chunks that put a border next to almost every word - plus a `--top` check.
110 checks in total. (We also verified the tests are meaningful: deliberately breaking the border rule makes 54 of them fail.)

## 6. Project layout

```
src/common.hpp      tokenizer, ownership rule (scan_words), hash map helpers, output format
src/wc_serial.cpp   sequential baseline
src/wc_mpi.cpp      hybrid MPI + OpenMP implementation
tests/              run_tests.sh + reference.py (independent oracle)
scripts/            gen_corpus.py, benchmark.py, plot.py
data/sample.txt     small example input
results/            generated by benchmark.py / plot.py
```

## 7. References

* MPI Forum, *MPI: A Message-Passing Interface Standard* (MPI-IO, `MPI_Alltoallv`, `MPI_Gatherv`)
* OpenMP Architecture Review Board, *OpenMP API Specification* (dynamic scheduling, thread-private data)
* FNV-1a hash function (public domain): http://www.isthe.com/chongo/tech/comp/fnv/
* No external code is copied into this repository; only the C++ standard library, MPI and OpenMP are used.
