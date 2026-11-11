# Demo guide: MPI in 10 minutes + questions you may get

You must be able to explain every part of the code during the demo. This page is a study sheet.
Read it together with `src/wc_mpi.cpp` (the comments there follow the same order).

## 1. MPI crash course (only what this project uses)

* `mpirun -np 4 ./wc_mpi ...` starts **4 independent copies** of the same program (called *ranks*, numbered 0..3).
  They do **not** share memory. They only exchange data through MPI calls.
* `MPI_Comm_rank` = "which copy am I?", `MPI_Comm_size` = "how many copies are there?".
* **MPI-IO** (`MPI_File_open`, `MPI_File_read_at`): every rank reads *its own* part of the same file at an offset.
* **`MPI_Alltoall` / `MPI_Alltoallv`**: every rank sends a (different) message to every other rank - used to ship
  each word to its *owner* rank. `v` = the messages may have different sizes (we pass sizes and displacements).
* **`MPI_Gatherv`**: everyone sends a list to rank 0. **`MPI_Reduce`**: combine numbers (sum / max) on rank 0.
* **`MPI_Barrier`**: wait until all ranks reach this line (used before starting the clock).
* `MPI_Init_thread(..., MPI_THREAD_FUNNELED)`: "I use threads, but only the main thread calls MPI".

## 2. OpenMP crash course (only what this project uses)

* `#pragma omp parallel` starts a team of threads that share the memory of ONE rank.
* `#pragma omp for schedule(dynamic, 1)`: loop iterations (our 1 MiB chunks) are handed out to whichever thread is free.
* Each thread writes only to `local[t]` (its own hash map) -> no data race, no lock.

## 3. The one idea to remember

> **A word belongs to the chunk that contains its first byte.**

`scan_words()` in `src/common.hpp` implements it (skip the word we start in the middle of; finish the last word even if it
runs past the end). It is used for ranks *and* for threads. The tests with 1-byte chunks and a 200 KB word prove it.

## 4. Live demo script (about 3-4 minutes)

```bash
make && make test                                   # builds, shows "passed: 110 failed: 0"
python3 scripts/gen_corpus.py data/demo.txt --size-mb 256
./build/wc_serial --top 5 data/demo.txt             # note the TIMING line (stderr)
mpirun -np 4 ./build/wc_mpi --threads 2 --top 5 data/demo.txt   # same output, shorter time
./build/wc_serial --top 5 data/demo.txt > a.txt 2>/dev/null
mpirun -np 4 ./build/wc_mpi --threads 2 --top 5 data/demo.txt > b.txt 2>/dev/null
diff a.txt b.txt && echo IDENTICAL
```
Add `--oversubscribe` / `--bind-to none` to `mpirun` if needed (see README). Show `results/*.png` afterwards.

## 5. Questions the instructor may ask (prepare answers)

1. *Why MPI **and** OpenMP?* MPI scales across processes and nodes and gives each rank its own memory; OpenMP shares one rank's
   memory between cores with less overhead and less duplicated data than one MPI process per core.
2. *What if a word is cut at a border?* The ownership rule (section 3).
3. *Why thread-private maps?* No locks in the hot loop; merging costs only O(distinct words).
4. *Why hash-partition the words instead of sending everything to rank 0?* Rank 0 would become a memory/network bottleneck.
5. *Why is the global top-N correct if each rank only sends its local top-N?* Every word lives on exactly one rank after the exchange, so
   the global top-N words must be in the top-N of their owner.
6. *Why is speedup below ideal?* I/O (disk/cache bandwidth), the sequential merge/sort, the all-to-all exchange, memory bandwidth, load imbalance
   (see the `breakdown.png` graph).
7. *Strong vs weak scaling?* Strong: fixed input, more cores. Weak: input grows with cores.
8. *How do you know it is correct?* 110 automatic checks against an independent Python reference (and a mutation check).
9. *Limitations?* Sequential thread-map merge, ASCII-only case folding, simple tokenizer (`don't` = 2 words).
