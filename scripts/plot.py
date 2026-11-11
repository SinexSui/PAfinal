#!/usr/bin/env python3
"""Turns results/strong.csv and results/weak.csv into the graphs for the report.

Definitions (T_serial = time of the sequential program, T(c) = parallel time on c cores):
    speedup            S(c) = T_serial / T(c)
    efficiency         E(c) = S(c) / c                (1.0 = perfect)
    weak-scaling eff.  W(c) = T_serial(base size) / T(c, size = c x base)   (1.0 = perfect)

Each point is the MEDIAN of the repetitions.

Usage: python3 scripts/plot.py          (reads results/*.csv, writes results/*.png and results/summary.md)
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
STYLE = {"mpi": ("o-", "Pure MPI (c ranks x 1 thread)"),
         "omp": ("s-", "Pure OpenMP (1 rank x c threads)"),
         "hybrid": ("^-", "Hybrid (2 ranks x c/2 threads)")}


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def medians(rows, key):
    """{(config, cores): median(key)} over repetitions."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["config"], int(r["cores"]))].append(float(r[key]))
    return {k: statistics.median(v) for k, v in groups.items()}


def strong(rows, summary):
    total = medians(rows, "total")
    t_serial = total[("serial", 1)]
    cores = sorted({c for (cfg, c) in total if cfg != "serial"})
    size = rows[0]["size_mb"]

    for metric, fname, ylabel in (("speedup", "speedup.png", "Speedup  T_serial / T(c)"),
                                  ("efficiency", "efficiency.png", "Efficiency  speedup / c")):
        plt.figure(figsize=(6.5, 4.5))
        for cfg, (fmt, label) in STYLE.items():
            pts = [(c, t_serial / total[(cfg, c)]) for c in cores if (cfg, c) in total]
            if not pts:
                continue
            xs, ys = zip(*pts)
            plt.plot(xs, [y / x if metric == "efficiency" else y for x, y in pts], fmt, label=label)
        ideal = [1.0 if metric == "efficiency" else c for c in cores]
        plt.plot(cores, ideal, "k--", label="Ideal")
        plt.xlabel("Cores (MPI ranks x OpenMP threads)")
        plt.ylabel(ylabel)
        plt.title(f"Strong scaling ({size} MB input)")
        if metric == "speedup":
            plt.xscale("log", base=2)
            plt.yscale("log", base=2)
        else:
            plt.xscale("log", base=2)
        plt.xticks(cores, [str(c) for c in cores])
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(RES, fname), dpi=150)
        plt.close()

    # time breakdown at the largest core count
    cmax = max(cores)
    labels, parts = ["serial"], [[medians(rows, k)[("serial", 1)] for k in ("read", "count", "exchange")]]
    for cfg in STYLE:
        if (cfg, cmax) in total:
            labels.append(f"{cfg}\n{cmax} cores")
            parts.append([medians(rows, k)[(cfg, cmax)] for k in ("read", "count", "exchange")])
    plt.figure(figsize=(6.5, 4.5))
    bottom = [0.0] * len(labels)
    for i, name in enumerate(("read (I/O)", "count (OpenMP)", "exchange + top-N (MPI)")):
        vals = [p[i] for p in parts]
        plt.bar(labels, vals, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.ylabel("Time (s)")
    plt.title("Where the time goes")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "breakdown.png"), dpi=150)
    plt.close()

    summary.append(f"## Strong scaling ({size} MB input, serial = {t_serial:.3f} s)\n")
    summary.append("| config | cores | time (s) | speedup | efficiency |")
    summary.append("|---|---|---|---|---|")
    for cfg in STYLE:
        for c in cores:
            if (cfg, c) in total:
                s = t_serial / total[(cfg, c)]
                summary.append(f"| {cfg} | {c} | {total[(cfg, c)]:.3f} | {s:.2f} | {s / c:.2f} |")
    summary.append("")


def weak(rows, summary):
    total = medians(rows, "total")
    t_base = total[("serial", 1)]
    cores = sorted({c for (cfg, c) in total if cfg != "serial"})
    plt.figure(figsize=(6.5, 4.5))
    for cfg, (fmt, label) in STYLE.items():
        pts = [(c, t_base / total[(cfg, c)]) for c in cores if (cfg, c) in total]
        if pts:
            xs, ys = zip(*pts)
            plt.plot(xs, ys, fmt, label=label)
    plt.plot(cores, [1.0] * len(cores), "k--", label="Ideal")
    plt.xscale("log", base=2)
    plt.xticks(cores, [str(c) for c in cores])
    plt.ylim(0, 1.3)
    plt.xlabel("Cores (input size grows proportionally)")
    plt.ylabel("Weak-scaling efficiency  T_serial(base) / T(c)")
    plt.title("Weak scaling")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "weak_scaling.png"), dpi=150)
    plt.close()

    summary.append(f"## Weak scaling (base size {rows[0]['size_mb']} MB per core, serial base = {t_base:.3f} s)\n")
    summary.append("| config | cores | time (s) | weak efficiency |")
    summary.append("|---|---|---|---|")
    for cfg in STYLE:
        for c in cores:
            if (cfg, c) in total:
                summary.append(f"| {cfg} | {c} | {total[(cfg, c)]:.3f} | {t_base / total[(cfg, c)]:.2f} |")
    summary.append("")


def main():
    summary = []
    s, w = load("strong.csv"), load("weak.csv")
    if s is None and w is None:
        sys.exit("no results found - run scripts/benchmark.py first")
    if s:
        strong(s, summary)
    if w:
        weak(w, summary)
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write("\n".join(summary))
    print("\n".join(summary))
    print("graphs written to", RES)


if __name__ == "__main__":
    main()
