#!/usr/bin/env python3
"""Benchmark driver: runs the serial and the MPI+OpenMP programs and writes a CSV.

Strong scaling:  the input size is FIXED, the number of cores grows.
                 -> "how much faster do I get on the same problem?"
Weak scaling:    the input size grows together with the number of cores
                 (size = base size x cores).
                 -> "can I process c times more data in the same time?"

For every core count c we measure three kinds of configurations:
    mpi     c ranks   x 1 thread          (pure MPI)
    omp     1 rank    x c threads         (pure OpenMP)
    hybrid  2 ranks   x c/2 threads       (MPI between 2 ranks, OpenMP inside each; c >= 4)

Examples:
    python3 scripts/benchmark.py --mode strong --size-mb 256 --max-cores 8
    python3 scripts/benchmark.py --mode weak --size-mb 64 --max-cores 8
    python3 scripts/benchmark.py --mode strong --file my_book_collection.txt
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIAL = os.path.join(ROOT, "build", "wc_serial")
PAR = os.path.join(ROOT, "build", "wc_mpi")
FIELDS = ["mode", "config", "procs", "threads", "cores", "size_mb", "rep",
          "total", "read", "count", "exchange"]


def mpirun_prefix(procs):
    """mpirun command line; adds the flags Open MPI needs for hybrid runs."""
    cmd = ["mpirun", "-np", str(procs)]
    try:
        version = subprocess.run(["mpirun", "--version"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        sys.exit("mpirun not found - install an MPI implementation (see README)")
    if "open mpi" in version.lower() or "open-mpi" in version.lower():
        # Without --bind-to none Open MPI pins all threads of a rank to ONE core.
        cmd += ["--bind-to", "none", "--oversubscribe"]
        if os.geteuid() == 0:
            cmd += ["--allow-run-as-root"]
    return cmd


def parse_timing(stderr):
    line = [l for l in stderr.splitlines() if l.startswith("TIMING")][-1]
    return {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", line)}


def run_once(procs, threads, path, is_serial):
    if is_serial:
        cmd = [SERIAL, "--top", "20", path]
    else:
        cmd = mpirun_prefix(procs) + [PAR, "--threads", str(threads), "--top", "20", path]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="spread", OMP_PLACES="cores")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"run failed: {' '.join(cmd)}\n{r.stderr}")
    return parse_timing(r.stderr)


def configs_for(c):
    out = [("mpi", c, 1)]
    if c > 1:
        out.append(("omp", 1, c))
    if c >= 4 and c % 2 == 0:
        out.append(("hybrid", 2, c // 2))
    return out


def core_list(max_cores):
    cores, c = [], 1
    while c <= max_cores:
        cores.append(c)
        c *= 2
    if cores[-1] != max_cores:
        cores.append(max_cores)
    return cores


def ensure_corpus(path, size_mb):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"generating {path} ({size_mb} MB) ...", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen_corpus.py"), path,
                    "--size-mb", str(size_mb)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["strong", "weak"], default="strong")
    ap.add_argument("--size-mb", type=float, default=256,
                    help="strong: total size; weak: size per core")
    ap.add_argument("--max-cores", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--reps", type=int, default=3, help="repetitions per configuration (median is used)")
    ap.add_argument("--file", help="strong scaling: use this text file instead of a generated one")
    ap.add_argument("--out", help="CSV output path (default results/<mode>.csv)")
    a = ap.parse_args()

    for exe in (SERIAL, PAR):
        if not os.path.exists(exe):
            sys.exit("build first: run `make`")
    out = a.out or os.path.join(ROOT, "results", f"{a.mode}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = []

    def record(config, procs, threads, cores, size_mb, path, is_serial=False):
        run_once(procs, threads, path, is_serial)            # warm-up (fills the OS file cache), discarded
        for rep in range(a.reps):
            t = run_once(procs, threads, path, is_serial)
            rows.append(dict(mode=a.mode, config=config, procs=procs, threads=threads, cores=cores,
                             size_mb=round(size_mb, 1), rep=rep, total=t["total"], read=t["read"],
                             count=t["count"], exchange=t["exchange"]))
        med = statistics.median(r["total"] for r in rows if r["config"] == config
                                and r["cores"] == cores and r["size_mb"] == round(size_mb, 1))
        print(f"{a.mode:6s} {config:7s} P={procs:<3d} T={threads:<3d} size={size_mb:7.1f} MB  median total={med:.3f} s",
              flush=True)

    if a.mode == "strong":
        path = a.file or os.path.join(ROOT, "data", "strong.txt")
        if not a.file:
            ensure_corpus(path, a.size_mb)
        size_mb = os.path.getsize(path) / 1024 / 1024
        record("serial", 1, 1, 1, size_mb, path, is_serial=True)
        for c in core_list(a.max_cores):
            for name, p, t in configs_for(c):
                record(name, p, t, c, size_mb, path)
    else:  # weak scaling
        base = os.path.join(ROOT, "data", "weak_1.txt")
        ensure_corpus(base, a.size_mb)
        record("serial", 1, 1, 1, os.path.getsize(base) / 1024 / 1024, base, is_serial=True)
        for c in core_list(a.max_cores):
            path = os.path.join(ROOT, "data", f"weak_{c}.txt")
            ensure_corpus(path, a.size_mb * c)
            size_mb = os.path.getsize(path) / 1024 / 1024
            for name, p, t in configs_for(c):
                record(name, p, t, c, size_mb, path)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
