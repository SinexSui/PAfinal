// wc_mpi.cpp - hybrid MPI + OpenMP parallel word count.
//
// Usage:  mpirun -np P ./wc_mpi [--threads T] [--top N] [--chunk BYTES] file
//
// Overview of the algorithm (see README.md for diagrams and the reasoning):
//
//   Phase 1  READ      Every MPI rank reads its own byte range of the file
//                      using MPI-IO (no rank ever holds the whole file).
//   Phase 2  COUNT     The rank splits its range into small chunks; OpenMP
//                      threads grab chunks dynamically and count words into
//                      thread-private hash maps (no locks, no sharing).
//                      Thread maps are then merged into one map per rank.
//   Phase 3  EXCHANGE  EXPLICIT DATA OWNERSHIP: every distinct word is owned
//                      by exactly one rank, owner = hash(word) % P. Ranks send
//                      each (word, count) pair to its owner with MPI_Alltoallv;
//                      the owner adds the partial counts together.
//   Phase 4  TOP-N     Each rank keeps its N best words (the global top N must
//                      be inside the union of these lists, because each word
//                      has exactly one owner). Rank 0 gathers and prints.
//
// Process/thread roles:  MPI = distributed memory between ranks,
//                        OpenMP = shared memory between the threads of a rank.
// MPI calls are made only outside OpenMP parallel regions -> MPI_THREAD_FUNNELED.

#include <mpi.h>
#include <omp.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.hpp"

// ----------------------------------------------------------------------------
// small helpers
// ----------------------------------------------------------------------------
[[noreturn]] static void die(const char* msg) {
    std::fprintf(stderr, "wc_mpi: fatal error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);  // not reached
}

// FNV-1a 64-bit hash. We use our own function (not std::hash) so that the
// owner of a word is guaranteed to be identical on every rank.
static uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

// MPI-IO read of exactly n bytes at `off` (MPI counts are `int`, so we loop).
static void read_at(MPI_File fh, MPI_Offset off, char* dst, size_t n) {
    while (n > 0) {
        int c = static_cast<int>(std::min<size_t>(n, 1u << 30));
        MPI_Status st;
        if (MPI_File_read_at(fh, off, dst, c, MPI_BYTE, &st) != MPI_SUCCESS) die("MPI_File_read_at failed");
        int got = 0;
        MPI_Get_count(&st, MPI_BYTE, &got);
        if (got <= 0) die("unexpected end of file while reading");
        off += got; dst += got; n -= static_cast<size_t>(got);
    }
}

// Wire format of one entry: [uint64 count][uint32 length][length bytes of the word]
static void append_entry(std::vector<char>& out, const std::string& w, uint64_t count) {
    uint32_t len = static_cast<uint32_t>(w.size());
    size_t pos = out.size();
    out.resize(pos + sizeof(count) + sizeof(len) + len);
    std::memcpy(&out[pos], &count, sizeof(count));
    std::memcpy(&out[pos + sizeof(count)], &len, sizeof(len));
    std::memcpy(&out[pos + sizeof(count) + sizeof(len)], w.data(), len);
}

template <class F>
static void for_each_entry(const char* p, size_t n, F&& f) {
    size_t pos = 0;
    while (pos < n) {
        uint64_t count; uint32_t len;
        std::memcpy(&count, p + pos, sizeof(count));
        std::memcpy(&len, p + pos + sizeof(count), sizeof(len));
        f(std::string(p + pos + sizeof(count) + sizeof(len), len), count);
        pos += sizeof(count) + sizeof(len) + len;
    }
}

// Turns per-rank byte counts into displacements, checking the int limit of MPI.
static std::vector<int> displacements(const std::vector<int>& counts) {
    std::vector<int> d(counts.size(), 0);
    long long sum = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        d[i] = static_cast<int>(sum);
        sum += counts[i];
        if (sum > INT_MAX) die("message too large for MPI (use more ranks)");
    }
    return d;
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    Options opt = parse_options(argc, argv);
    if (!opt.ok) {
        if (rank == 0) std::fprintf(stderr, "usage: %s [--threads T] [--top N] [--chunk BYTES] file\n", argv[0]);
        MPI_Finalize();
        return 2;
    }
    if (opt.threads > 0) omp_set_num_threads(opt.threads);
    const int T = omp_get_max_threads();

    // ---- open the file and compute this rank's byte range -------------------
    MPI_File fh;
    if (MPI_File_open(MPI_COMM_WORLD, opt.file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
        die("cannot open input file");
    MPI_Offset fsize = 0;
    MPI_File_get_size(fh, &fsize);

    // Block distribution: ranges differ by at most one byte.
    const MPI_Offset base = fsize / P, rem = fsize % P;
    const MPI_Offset start = rank * base + std::min<MPI_Offset>(rank, rem);
    const MPI_Offset end = start + base + (rank < rem ? 1 : 0);
    const size_t own_len = static_cast<size_t>(end - start);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    // ---- Phase 1: READ -------------------------------------------------------
    // Buffer layout:  [1 byte before start (if any)] [owned bytes] [tail]
    //  * the byte before `start` tells us whether the first word of our range
    //    began in the previous rank's range (then it is not ours)
    //  * the tail lets us finish the last word that STARTS in our range
    const size_t prefix = (start > 0) ? 1 : 0;
    std::vector<char> buf(prefix + own_len);
    if (!buf.empty()) read_at(fh, start - static_cast<MPI_Offset>(prefix), buf.data(), buf.size());

    if (own_len > 0 && end < fsize && is_word_byte(static_cast<unsigned char>(buf.back()))) {
        const size_t BLOCK = 1u << 16;
        MPI_Offset pos = end;
        while (pos < fsize) {
            size_t n = static_cast<size_t>(std::min<MPI_Offset>(BLOCK, fsize - pos));
            size_t old = buf.size();
            buf.resize(old + n);
            read_at(fh, pos, buf.data() + old, n);
            pos += static_cast<MPI_Offset>(n);
            bool found_separator = false;
            for (size_t k = old; k < buf.size(); ++k)
                if (!is_word_byte(static_cast<unsigned char>(buf[k]))) { found_separator = true; break; }
            if (found_separator) break;  // the last word is complete
        }
    }
    MPI_File_close(&fh);
    const double t1 = MPI_Wtime();

    // ---- Phase 2: COUNT (OpenMP) --------------------------------------------
    const size_t owned_begin = prefix, owned_end = prefix + own_len, len = buf.size();
    const long nchunks = static_cast<long>((own_len + opt.chunk - 1) / opt.chunk);
    std::vector<CountMap> local(T);
    std::vector<uint64_t> local_total(T, 0);

#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        CountMap& my_map = local[t];  // thread-private table: no locks needed
        std::string tmp;
        uint64_t my_total = 0;
#pragma omp for schedule(dynamic, 1)
        for (long c = 0; c < nchunks; ++c) {
            size_t b = owned_begin + static_cast<size_t>(c) * opt.chunk;
            size_t e = std::min(b + opt.chunk, owned_end);
            scan_words(buf.data(), len, b, e, [&](const char* w, size_t n) {
                count_word(my_map, tmp, w, n);
                ++my_total;
            });
        }
        local_total[t] = my_total;
    }
    std::vector<char>().swap(buf);  // free the text, we only need the counts now

    // Merge the thread tables into one table for this rank.
    CountMap rank_map = std::move(local[0]);
    for (int t = 1; t < T; ++t) {
        for (const auto& kv : local[t]) rank_map[kv.first] += kv.second;
        CountMap().swap(local[t]);
    }
    uint64_t my_words = 0;
    for (int t = 0; t < T; ++t) my_words += local_total[t];
    const double t2 = MPI_Wtime();

    // ---- Phase 3: EXCHANGE - every word goes to its owner rank ---------------
    std::vector<std::vector<char>> outgoing(P);
    for (const auto& kv : rank_map) append_entry(outgoing[fnv1a(kv.first) % P], kv.first, kv.second);
    CountMap().swap(rank_map);

    std::vector<int> send_counts(P), recv_counts(P);
    for (int r = 0; r < P; ++r) send_counts[r] = static_cast<int>(outgoing[r].size());
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> send_displs = displacements(send_counts), recv_displs = displacements(recv_counts);
    std::vector<char> send_buf, recv_buf;
    send_buf.reserve(static_cast<size_t>(send_displs.back()) + send_counts.back());
    for (int r = 0; r < P; ++r) {
        send_buf.insert(send_buf.end(), outgoing[r].begin(), outgoing[r].end());
        std::vector<char>().swap(outgoing[r]);
    }
    recv_buf.resize(static_cast<size_t>(recv_displs.back()) + recv_counts.back());
    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(), MPI_BYTE,
                  recv_buf.data(), recv_counts.data(), recv_displs.data(), MPI_BYTE, MPI_COMM_WORLD);

    // We now own a disjoint subset of the vocabulary; add up the partial counts.
    CountMap owned;
    for_each_entry(recv_buf.data(), recv_buf.size(), [&](std::string w, uint64_t c) { owned[std::move(w)] += c; });
    std::vector<char>().swap(recv_buf);
    std::vector<char>().swap(send_buf);

    // ---- Phase 4: TOP-N ------------------------------------------------------
    std::vector<WordCount> candidates(owned.begin(), owned.end());
    const uint64_t my_unique = candidates.size();
    sort_top(candidates, opt.top_n);

    std::vector<char> mine;
    for (const auto& kv : candidates) append_entry(mine, kv.first, kv.second);
    int my_bytes = static_cast<int>(mine.size());
    if (mine.size() > INT_MAX) die("top-N list too large");

    std::vector<int> gather_counts(P), gather_displs;
    MPI_Gather(&my_bytes, 1, MPI_INT, gather_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> gathered;
    if (rank == 0) {
        gather_displs = displacements(gather_counts);
        gathered.resize(static_cast<size_t>(gather_displs.back()) + gather_counts.back());
    }
    MPI_Gatherv(mine.data(), my_bytes, MPI_BYTE, gathered.data(), gather_counts.data(),
                gather_displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    unsigned long long total_words = 0, unique_words = 0;
    unsigned long long a = my_words, b = my_unique;
    MPI_Reduce(&a, &total_words, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&b, &unique_words, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    const double t3 = MPI_Wtime();

    // ---- timing: the slowest rank defines each phase -------------------------
    double mine_t[4] = {t1 - t0, t2 - t1, t3 - t2, t3 - t0}, max_t[4] = {0, 0, 0, 0};
    MPI_Reduce(mine_t, max_t, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<WordCount> result;
        for_each_entry(gathered.data(), gathered.size(),
                       [&](std::string w, uint64_t c) { result.emplace_back(std::move(w), c); });
        sort_top(result, opt.top_n);
        print_result(total_words, unique_words, result);
        std::fprintf(stderr, "TIMING total=%.6f read=%.6f count=%.6f exchange=%.6f ranks=%d threads=%d\n",
                     max_t[3], max_t[0], max_t[1], max_t[2], P, T);
    }
    MPI_Finalize();
    return 0;
}
