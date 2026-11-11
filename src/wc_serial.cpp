// wc_serial.cpp - sequential baseline.
//
// Reads the whole file into memory, counts every word in one pass with a hash
// map, then sorts the result. This is the reference for correctness AND the
// denominator of every speedup number.
//
// Usage: ./wc_serial [--top N] file      (N = 0 prints all words)

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>

#include "common.hpp"

int main(int argc, char** argv) {
    Options opt = parse_options(argc, argv);
    if (!opt.ok) {
        std::fprintf(stderr, "usage: %s [--top N] file\n", argv[0]);
        return 2;
    }
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    // 1) read the file
    std::ifstream in(opt.file, std::ios::binary);
    if (!in) { std::fprintf(stderr, "cannot open %s\n", opt.file.c_str()); return 1; }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto t1 = clk::now();

    // 2) count
    CountMap map;
    std::string tmp;
    uint64_t total = 0;
    scan_words(text.data(), text.size(), 0, text.size(), [&](const char* w, size_t n) {
        count_word(map, tmp, w, n);
        ++total;
    });
    auto t2 = clk::now();

    // 3) sort and print
    std::vector<WordCount> all(map.begin(), map.end());
    uint64_t unique = all.size();
    sort_top(all, opt.top_n);
    auto t3 = clk::now();
    print_result(total, unique, all);

    auto sec = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    std::fprintf(stderr, "TIMING total=%.6f read=%.6f count=%.6f exchange=%.6f ranks=1 threads=1\n",
                 sec(t0, t3), sec(t0, t1), sec(t1, t2), sec(t2, t3));
    return 0;
}
