// common.hpp - code shared by the serial and the parallel (MPI + OpenMP) programs.
//
// Keeping the tokenizer and the output logic in ONE place guarantees that the
// serial and parallel versions use exactly the same definition of "a word".
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Word definition
// ---------------------------------------------------------------------------
// A word is a maximal run of "word bytes":
//   * ASCII letters and digits
//   * any byte >= 0x80 (so UTF-8 text such as Cyrillic is kept intact)
// Everything else (space, punctuation, '\n', '\r', '-', '\'' ...) is a separator.
// ASCII letters are lower-cased, so "The" and "the" are the same word.
// (Non-ASCII letters are NOT case-folded - documented limitation.)
inline bool is_word_byte(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') || c >= 0x80;
}

inline char to_lower_ascii(unsigned char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : static_cast<char>(c);
}

// ---------------------------------------------------------------------------
// The ownership rule (the key idea that makes chunking correct)
// ---------------------------------------------------------------------------
// When a text is cut into chunks, a word can straddle a cut. We avoid double
// counting with ONE simple rule:
//
//     A word belongs to the chunk that contains its FIRST byte.
//
// scan_words() processes the "owned" range [begin, end) of a buffer:
//   * a word that started BEFORE `begin` is skipped (the previous chunk owns it)
//   * a word that starts inside [begin, end) is counted completely, even if it
//     runs past `end` (so the buffer must contain the bytes after `end`
//     until the word finishes).
// For begin > 0 the byte buf[begin-1] must be available to tell whether we are
// in the middle of a word.
template <class Emit>
void scan_words(const char* buf, size_t len, size_t begin, size_t end, Emit&& emit) {
    size_t i = begin;
    if (begin > 0 && is_word_byte(static_cast<unsigned char>(buf[begin - 1]))) {
        // We are inside a word owned by the previous chunk: skip its remainder.
        while (i < len && is_word_byte(static_cast<unsigned char>(buf[i]))) ++i;
    }
    while (i < end) {
        if (!is_word_byte(static_cast<unsigned char>(buf[i]))) { ++i; continue; }
        size_t j = i;
        while (j < len && is_word_byte(static_cast<unsigned char>(buf[j]))) ++j;
        emit(buf + i, j - i);  // word = buf[i .. j)
        i = j;
    }
}

// ---------------------------------------------------------------------------
// Counting table
// ---------------------------------------------------------------------------
using CountMap = std::unordered_map<std::string, uint64_t>;
using WordCount = std::pair<std::string, uint64_t>;

// Adds one occurrence of the (lower-cased) word w[0..n) to the map.
// `tmp` is a scratch string reused between calls to avoid allocations.
inline void count_word(CountMap& map, std::string& tmp, const char* w, size_t n) {
    tmp.resize(n);
    for (size_t k = 0; k < n; ++k) tmp[k] = to_lower_ascii(static_cast<unsigned char>(w[k]));
    auto it = map.find(tmp);
    if (it == map.end()) map.emplace(tmp, 1);
    else ++it->second;
}

// Deterministic ordering: higher count first, ties broken alphabetically.
// A total order is what makes serial/parallel output byte-for-byte comparable.
inline bool better(const WordCount& a, const WordCount& b) {
    if (a.second != b.second) return a.second > b.second;
    return a.first < b.first;
}

// Keeps only the `top_n` best entries (top_n == 0 means "keep everything") and sorts them.
inline void sort_top(std::vector<WordCount>& v, size_t top_n) {
    if (top_n == 0 || top_n >= v.size()) {
        std::sort(v.begin(), v.end(), better);
    } else {
        std::partial_sort(v.begin(), v.begin() + top_n, v.end(), better);
        v.resize(top_n);
    }
}

// Prints the result in a fixed, easily diff-able format.
inline void print_result(uint64_t total, uint64_t unique, const std::vector<WordCount>& top) {
    std::printf("TOTAL_WORDS %llu\n", static_cast<unsigned long long>(total));
    std::printf("UNIQUE_WORDS %llu\n", static_cast<unsigned long long>(unique));
    for (const auto& kv : top)
        std::printf("%s\t%llu\n", kv.first.c_str(), static_cast<unsigned long long>(kv.second));
}

// ---------------------------------------------------------------------------
// Tiny command line parser shared by both programs
// ---------------------------------------------------------------------------
struct Options {
    std::string file;
    size_t top_n = 20;        // --top N   (0 = print every word)
    int threads = 0;          // --threads T (0 = use OMP_NUM_THREADS / default)
    size_t chunk = 1u << 20;  // --chunk BYTES: OpenMP work unit (tests use tiny values)
    bool ok = false;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--top" && i + 1 < argc) o.top_n = std::stoull(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) o.threads = std::stoi(argv[++i]);
        else if (a == "--chunk" && i + 1 < argc) o.chunk = std::stoull(argv[++i]);
        else if (!a.empty() && a[0] != '-') o.file = a;
        else return o;  // unknown option -> ok stays false
    }
    o.ok = !o.file.empty() && o.chunk > 0;
    return o;
}
