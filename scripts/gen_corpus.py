#!/usr/bin/env python3
"""Generate a synthetic English-like text file for benchmarking.

Words are drawn from a Zipf distribution (a few words are very frequent, most are
rare), which is what real natural-language text looks like. Capitalisation,
punctuation and line breaks are mixed in so the tokenizer is exercised properly.

Usage: python3 scripts/gen_corpus.py OUTPUT_FILE --size-mb 256 [--vocab 50000] [--seed 1]

To generate quickly we build a few different ~4 MB blocks and repeat them in a
cycle until the requested size is reached.
"""
import argparse
import random

import numpy as np

SYLLABLES = ["ka", "to", "mi", "ra", "ne", "so", "lu", "vi", "pa", "de", "or", "an",
             "el", "is", "um", "ta", "re", "no", "ba", "li", "gu", "ho", "fa", "zi"]


def build_vocab(n, rng):
    words = set()
    while len(words) < n:
        words.add("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(1, 4))))
    return sorted(words)


def make_block(vocab, weights, rng_np, rng, n_words):
    idx = rng_np.choice(len(vocab), size=n_words, p=weights)
    puncts = ["", "", "", "", ",", ".", ";", " -", "!"]
    out = []
    for k, i in enumerate(idx):
        w = vocab[i]
        if rng.random() < 0.08:
            w = w.capitalize()
        out.append(w)
        out.append(rng.choice(puncts))
        out.append("\n" if k % 12 == 11 else " ")
    return "".join(out).encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--size-mb", type=float, default=64)
    ap.add_argument("--vocab", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rng_np = np.random.default_rng(a.seed)
    vocab = build_vocab(a.vocab, rng)
    ranks = np.arange(1, len(vocab) + 1, dtype=np.float64)
    weights = 1.0 / ranks            # Zipf's law: frequency ~ 1 / rank
    weights /= weights.sum()

    target = int(a.size_mb * 1024 * 1024)
    blocks = [make_block(vocab, weights, rng_np, rng, 600_000) for _ in range(4)]
    written = 0
    with open(a.output, "wb") as f:
        i = 0
        while written < target:
            chunk = blocks[i % len(blocks)][: target - written]
            f.write(chunk)
            written += len(chunk)
            i += 1
    print(f"wrote {written / 1024 / 1024:.1f} MB to {a.output}")


if __name__ == "__main__":
    main()
