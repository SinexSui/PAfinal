#!/usr/bin/env python3
"""Independent reference implementation used as the test oracle.

It is written with Python's `re` module and `collections.Counter`, so it shares
NO code with the C++ programs. It prints in the same format as the C++ programs.

Usage: python3 tests/reference.py [--top N] FILE      (N = 0 prints all words)
"""
import re
import sys
from collections import Counter

# word bytes: ASCII letters/digits and every byte >= 0x80
WORD = re.compile(rb"[0-9A-Za-z\x80-\xff]+")


def main():
    args = sys.argv[1:]
    top = 20
    if args and args[0] == "--top":
        top = int(args[1])
        args = args[2:]
    data = open(args[0], "rb").read()
    words = [w.lower() for w in WORD.findall(data)]   # bytes.lower() is ASCII-only
    counts = Counter(words)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if top:
        items = items[:top]
    out = [f"TOTAL_WORDS {len(words)}", f"UNIQUE_WORDS {len(counts)}"]
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()
    for w, c in items:
        sys.stdout.buffer.write(w + b"\t" + str(c).encode() + b"\n")


if __name__ == "__main__":
    main()
