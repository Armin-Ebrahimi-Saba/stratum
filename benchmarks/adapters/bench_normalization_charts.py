#!/usr/bin/env python3
"""
Normalization staged benchmark — combinatorial speedup and runtime charts.

Compares the Rust normalization kernels against their sklearn equivalents across
every (kernel, size, thread-count) combination. Analogous to
bench_elastic_net_charts.py, but there is no commit dimension — the kernels live
on the current HEAD, so nothing is checked out or rebuilt.

Kernels (grouped into families, selectable individually):
    l1               normalize(X, norm='l1')            [+ in-place]
    l2               normalize(X, norm='l2')            [+ in-place]
    max              normalize(X, norm='max')           [+ in-place]

Execution is fully stageable — data is collected and saved cell-by-cell, so a
partial run can be resumed without re-running completed work.

  Kernel-family flags (combine freely; default = all families):
    --l1  --l2  --max
    --no-inplace     drop the in-place kernel variants

  Size flags (independent; default = all tiers):
    --small    10k–1M rows
    --large    1M rows, wider
    --xlarge   2M–5M rows
    --xxlarge  10M rows (~2 GB f32)

  Thread flags:
    --threads 1,2,4,8    comma-separated subset (default: 1 … N_CPU)
    Maximum is fetched from os.cpu_count() — never hardcoded.

Correctness:
    Before timing, each selected kernel's Rust output is compared element-wise
    against sklearn's (rtol=1e-3, atol=1e-4) on the smaller size tiers. Results
    are cached under verify__<kernel>__<rows>__<cols>. Pass --no-verify to skip.

Other flags:
    --no-verify     skip the rust-vs-sklearn correctness check
    --plot-only     regenerate charts without re-running benchmarks
    --refresh       ignore cache and re-run everything
    --reps N        timing repetitions (default 5, auto-reduced for large inputs)

Cache: benchmarks/results/normalization_full_cache.json

Examples:
    uv run python benchmarks/adapters/bench_normalization_charts.py --small
    uv run python benchmarks/adapters/bench_normalization_charts.py --l1 --l2
    uv run python benchmarks/adapters/bench_normalization_charts.py --small --threads 1,4
    uv run python benchmarks/adapters/bench_normalization_charts.py --plot-only
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
RESULTS    = REPO_ROOT / "benchmarks" / "results"
CACHE_FILE = RESULTS / "normalization_full_cache.json"

# ── kernels ───────────────────────────────────────────────────────────────────
# (key, family, label, is_inplace)
KERNELS = [
    ("l2",         "l2",  "L2 (copy)",      False),
    ("l2_inplace", "l2",  "L2 (in-place)",  True),
    ("l1",         "l1",  "L1 (copy)",      False),
    ("l1_inplace", "l1",  "L1 (in-place)",  True),
    ("max",        "max", "Max (copy)",     False),
    ("max_inplace","max", "Max (in-place)", True),
]
KERNEL_LABEL  = {k: lbl for k, _, lbl, _ in KERNELS}
FAMILIES      = ["l1", "l2", "max"]

# ── thread counts — N_CPU fetched dynamically, never hardcoded ────────────────
_n_cpu      = os.cpu_count() or 4
ALL_THREADS = sorted(set(t for t in [1, 2, 4, 8, _n_cpu] if t <= _n_cpu))

# ── size tiers ────────────────────────────────────────────────────────────────
SIZE_SMALL   = [(10_000, 100), (100_000, 100), (100_000, 1_000), (1_000_000, 50)]
SIZE_LARGE   = [(1_000_000, 100), (1_000_000, 500)]
SIZE_XLARGE  = [(2_000_000, 100), (5_000_000, 50)]
SIZE_XXLARGE = [(10_000_000, 50)]

# ── benchmark config ──────────────────────────────────────────────────────────
N_REPS = 5

# Correctness check: Rust output vs sklearn output, element-wise.
VERIFY_RTOL   = 1e-3
VERIFY_ATOL   = 1e-4
# Verify only on inputs at or below this size (MB) — correctness is size
# independent, so there is no need to pay for the huge tiers.
VERIFY_MAX_MB = 200


# ─── shared timing helpers (used inside the worker subprocess) ────────────────

def _make_data(n_rows: int, n_cols: int, seed: int = 42):
    import numpy as np
    rng  = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    # ~1% all-zero rows to exercise the divide-by-zero guard, matching the
    # original bench_normalization.py data generator.
    zero_idx = rng.integers(0, n_rows, size=max(1, n_rows // 100))
    data[zero_idx] = 0.0
    return data


def _timeit(fn, n_reps: int) -> float:
    import numpy as np
    times = []
    for _ in range(n_reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _warmup(fn, n: int = 3):
    for _ in range(n):
        fn()
    gc.collect()


def _timeit_inplace(make_buf, fn, n_reps: int) -> float:
    """Rebuild the (mutated) buffer before each rep so the copy cost stays
    outside the timed region — matches the original benchmark's methodology."""
    import numpy as np
    for _ in range(3):
        fn(make_buf())
    gc.collect()
    times = []
    for _ in range(n_reps):
        buf = make_buf()
        gc.collect()
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ─── workers ───────────────────────────────────────────────────────────────────

def _worker_rust(kernel: str, r: int, c: int, n_reps: int):
    from stratum import _rust_backend as rb
    data = _make_data(r, c)

    if kernel in ("l1", "l2", "max"):
        fn = {"l1": rb.normalize_l1, "l2": rb.normalize_l2, "max": rb.normalize_max}[kernel]
        _warmup(lambda: fn(data)); t = _timeit(lambda: fn(data), n_reps)

    elif kernel in ("l1_inplace", "l2_inplace", "max_inplace"):
        fn = {"l1_inplace": rb.normalize_l1_inplace,
              "l2_inplace": rb.normalize_l2_inplace,
              "max_inplace": rb.normalize_max_inplace}[kernel]
        t = _timeit_inplace(lambda: data.copy(), fn, n_reps)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")

    print(json.dumps({"time": t}))


def _worker_sklearn(kernel: str, r: int, c: int, n_reps: int):
    from sklearn.preprocessing import normalize
    data = _make_data(r, c)

    if kernel in ("l1", "l2", "max"):
        _warmup(lambda: normalize(data, norm=kernel, copy=True))
        t = _timeit(lambda: normalize(data, norm=kernel, copy=True), n_reps)

    elif kernel in ("l1_inplace", "l2_inplace", "max_inplace"):
        norm = kernel.split("_")[0]
        t = _timeit_inplace(lambda: data.copy(),
                            lambda buf: normalize(buf, norm=norm, copy=False), n_reps)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")

    print(json.dumps({"time": t}))


def _rust_output(kernel, data, rb):
    """Materialise the Rust kernel's output array for a fresh copy of data."""
    if kernel in ("l1", "l2", "max"):
        fn = {"l1": rb.normalize_l1, "l2": rb.normalize_l2, "max": rb.normalize_max}[kernel]
        return fn(data)
    if kernel in ("l1_inplace", "l2_inplace", "max_inplace"):
        fn = {"l1_inplace": rb.normalize_l1_inplace,
              "l2_inplace": rb.normalize_l2_inplace,
              "max_inplace": rb.normalize_max_inplace}[kernel]
        buf = data.copy(); fn(buf); return buf
    raise ValueError(f"unknown kernel {kernel!r}")


def _sklearn_output(kernel, data):
    """The reference output sklearn produces for the same kernel."""
    from sklearn.preprocessing import normalize
    if kernel in ("l1", "l2", "max"):
        return normalize(data, norm=kernel, copy=True)
    if kernel in ("l1_inplace", "l2_inplace", "max_inplace"):
        return normalize(data, norm=kernel.split("_")[0], copy=True)
    raise ValueError(f"unknown kernel {kernel!r}")


def _worker_verify(kernel: str, r: int, c: int):
    import numpy as np
    from stratum import _rust_backend as rb
    data = _make_data(r, c)
    ru = np.asarray(_rust_output(kernel, data, rb), dtype=np.float64)
    sk = np.asarray(_sklearn_output(kernel, data), dtype=np.float64)
    diff    = np.abs(ru - sk)
    max_abs = float(diff.max())
    max_rel = float((diff / (np.abs(sk) + 1e-6)).max())
    passed  = bool(np.allclose(ru, sk, rtol=VERIFY_RTOL, atol=VERIFY_ATOL))
    print(json.dumps({"passed": passed, "max_abs_err": max_abs, "max_rel_err": max_rel}))


def _run_worker():
    a      = sys.argv
    mode   = a[a.index("--mode") + 1]
    kernel = a[a.index("--kernel") + 1]
    r      = int(a[a.index("--n-rows") + 1])
    c      = int(a[a.index("--n-cols") + 1])
    if mode == "verify":
        _worker_verify(kernel, r, c)
        return
    n_reps = int(a[a.index("--n-reps") + 1])
    if mode == "sklearn":
        _worker_sklearn(kernel, r, c, n_reps)
    else:
        _worker_rust(kernel, r, c, n_reps)


# ─── subprocess launcher ───────────────────────────────────────────────────────

def _spawn(kernel: str, r: int, c: int, mode: str,
           n_reps: int, n_threads: int | None = None) -> dict:
    env = dict(os.environ)
    env["SKRUB_RUST"] = "1"
    if n_threads is not None:
        env["SKRUB_RUST_THREADS"] = str(n_threads)
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker", "--mode", mode, "--kernel", kernel,
           "--n-rows", str(r), "--n-cols", str(c), "--n-reps", str(n_reps)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO_ROOT)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[-800:])
    return json.loads(p.stdout.strip())


# ─── cache ────────────────────────────────────────────────────────────────────

def _rkey(kernel: str, n_threads: int, r: int, c: int) -> str:
    return f"{kernel}__t{n_threads}__{r}__{c}"

def _skey(kernel: str, r: int, c: int) -> str:
    return f"sklearn__{kernel}__{r}__{c}"

def _vkey(kernel: str, r: int, c: int) -> str:
    return f"verify__{kernel}__{r}__{c}"

def load_cache() -> dict:
    return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}

def save_cache(cache: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def reps_for(r: int, c: int, base: int) -> int:
    mb = r * c * 4 / 1e6
    if mb > 500: return min(base, 3)
    if mb > 100: return min(base, 4)
    return base


# ─── plotting ─────────────────────────────────────────────────────────────────

def _size_label(r: int, c: int) -> str:
    mb = r * c * 4 / 1e6
    k = r // 1000
    rows = f"{k // 1000}M" if k >= 1000 else f"{k}k"
    return f"{rows}×{c}\n({mb:.0f} MB)"


def _ms_fmt(v, _):  return f"{v:.0f}" if v >= 1 else f"{v:.2f}"
def _sx_fmt(v, _):  return f"{v:.0f}×" if v >= 2 else f"{v:.1f}×"


def _kernel_colors(kernels):
    import matplotlib.pyplot as plt
    cmap = plt.cm.tab20
    return {k: cmap(i % 20) for i, k in enumerate(kernels)}


def _size_slug(r, c):
    return f"{r}x{c}"


def plot_groupA(cache, kernels, sizes, thread_counts, metric):
    """One file per thread count: x = sizes, one line per kernel.
    metric 'speedup' → sklearn/rust ; 'runtime' → rust ms (log)."""
    import matplotlib; matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    colors = _kernel_colors(kernels)
    xs     = list(range(len(sizes)))
    labels = [_size_label(r, c) for r, c in sizes]

    for n_t in thread_counts:
        plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 12,
                             "legend.fontsize": 8, "xtick.labelsize": 8})
        fig, ax = plt.subplots(figsize=(11, 6))
        title = "Speedup over sklearn" if metric == "speedup" else "Rust runtime (ms, log)"
        ax.set_title(f"{title} — {n_t} thread{'s' if n_t > 1 else ''}  "
                     f"[{_n_cpu} CPU cores]", fontsize=12, fontweight="bold")

        if metric == "speedup":
            ax.axhline(1.0, color="#888", lw=1.1, ls="--", label="sklearn (1×)", zorder=1)
        for k in kernels:
            ys, vx = [], []
            for j, (r, c) in enumerate(sizes):
                ru = cache.get(_rkey(k, n_t, r, c), {}).get("time")
                sk = cache.get(_skey(k, r, c), {}).get("time")
                if ru and ru > 0:
                    if metric == "speedup" and sk:
                        vx.append(j); ys.append(sk / ru)
                    elif metric == "runtime":
                        vx.append(j); ys.append(ru * 1000)
            if ys:
                ls = "--" if k.endswith("inplace") else "-"
                ax.plot(vx, ys, color=colors[k], lw=1.7, ls=ls, marker="o", ms=4,
                        label=KERNEL_LABEL[k], zorder=2)
        ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("Input size")
        if metric == "speedup":
            ax.set_ylabel("Speedup (×)"); ax.yaxis.set_major_formatter(mticker.FuncFormatter(_sx_fmt))
        else:
            ax.set_ylabel("Rust time (ms)"); ax.yaxis.set_major_formatter(mticker.FuncFormatter(_ms_fmt))
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(True, alpha=0.2, which="both")
        ax.set_xlim(-0.4, len(sizes) - 0.6)

        fig.tight_layout()
        out = RESULTS / f"normalization_groupA_{metric}_t{n_t}.pdf"
        fig.savefig(out, bbox_inches="tight"); print(f"  → {out.relative_to(REPO_ROOT)}")
        plt.close(fig)


def plot_groupB(cache, kernels, sizes, thread_counts, metric):
    """One file per size: x = thread count, one line per kernel."""
    import matplotlib; matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    colors = _kernel_colors(kernels)

    for (r, c) in sizes:
        plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 12,
                             "legend.fontsize": 8, "xtick.labelsize": 9})
        fig, ax = plt.subplots(figsize=(9, 6))
        title = "Speedup over sklearn" if metric == "speedup" else "Rust runtime (ms, log)"
        mb = r * c * 4 / 1e6
        ax.set_title(f"{title} — {r//1000}k×{c} ({mb:.0f} MB)  "
                     f"[{_n_cpu} CPU cores]", fontsize=12, fontweight="bold")

        if metric == "speedup":
            ax.axhline(1.0, color="#888", lw=1.1, ls="--", label="sklearn (1×)", zorder=1)
        for k in kernels:
            tx, ty = [], []
            for n_t in thread_counts:
                ru = cache.get(_rkey(k, n_t, r, c), {}).get("time")
                sk = cache.get(_skey(k, r, c), {}).get("time")
                if ru and ru > 0:
                    if metric == "speedup" and sk:
                        tx.append(n_t); ty.append(sk / ru)
                    elif metric == "runtime":
                        tx.append(n_t); ty.append(ru * 1000)
            if ty:
                ls = "--" if k.endswith("inplace") else "-"
                ax.plot(tx, ty, color=colors[k], lw=1.7, ls=ls, marker="o", ms=4,
                        label=KERNEL_LABEL[k], zorder=2)
        ax.set_yscale("log")
        ax.set_xticks(thread_counts)
        ax.set_xlabel("Thread count")
        if metric == "speedup":
            ax.set_ylabel("Speedup (×)"); ax.yaxis.set_major_formatter(mticker.FuncFormatter(_sx_fmt))
        else:
            ax.set_ylabel("Rust time (ms)"); ax.yaxis.set_major_formatter(mticker.FuncFormatter(_ms_fmt))
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(True, alpha=0.2, which="both")

        fig.tight_layout()
        out = RESULTS / f"normalization_groupB_{metric}_{_size_slug(r, c)}.pdf"
        fig.savefig(out, bbox_inches="tight"); print(f"  → {out.relative_to(REPO_ROOT)}")
        plt.close(fig)


# ─── benchmark runner ─────────────────────────────────────────────────────────

def run_benchmarks(cache, kernels, sizes, thread_counts, base_reps, no_verify=False):
    # ── correctness: Rust output vs sklearn (size-independent, small sizes) ──
    if not no_verify:
        verify_sizes = [s for s in sizes if s[0] * s[1] * 4 / 1e6 <= VERIFY_MAX_MB]
        if not verify_sizes and sizes:
            verify_sizes = [min(sizes, key=lambda s: s[0] * s[1])]
        pending = [(k, r, c) for k in kernels for (r, c) in verify_sizes
                   if _vkey(k, r, c) not in cache]
        if pending:
            print(f"── correctness check (rust vs sklearn, "
                  f"rtol={VERIFY_RTOL}, atol={VERIFY_ATOL}) ──")
        any_fail = False
        for kernel, r, c in pending:
            print(f"  {KERNEL_LABEL[kernel]:<22} {r:>10,}×{c} ...", end=" ", flush=True)
            try:
                v = _spawn(kernel, r, c, "verify", 1, n_threads=1)
                cache[_vkey(kernel, r, c)] = v
                save_cache(cache)
                st = "PASS ✓" if v["passed"] else "FAIL ✗"
                print(f"{st}  max_abs={v['max_abs_err']:.2e}  max_rel={v['max_rel_err']:.2e}")
                any_fail |= not v["passed"]
            except Exception as e:
                print(f"ERROR: {e}"); any_fail = True
        if any_fail:
            print("  WARNING: some kernels FAILED correctness — inspect the rows above "
                  "before trusting their timings.\n")
        elif pending:
            print("  all checked kernels PASS.\n")

    for kernel in kernels:
        # sklearn baseline (once per size; thread count does not apply)
        for r, c in sizes:
            if _skey(kernel, r, c) in cache:
                continue
            nr = reps_for(r, c, base_reps)
            tag = f"  sklearn  {KERNEL_LABEL[kernel]:<22} {r:>10,}×{c}"
            print(f"{tag} ...", end=" ", flush=True)
            try:
                res = _spawn(kernel, r, c, "sklearn", nr)
                cache[_skey(kernel, r, c)] = res
                save_cache(cache)
                print(f"{res['time']*1000:.1f} ms")
            except Exception as e:
                print(f"FAILED: {e}")

        # rust, per thread count
        for n_t in thread_counts:
            for r, c in sizes:
                if _rkey(kernel, n_t, r, c) in cache:
                    continue
                nr = reps_for(r, c, base_reps)
                tag = f"  rust {n_t}T  {KERNEL_LABEL[kernel]:<22} {r:>10,}×{c}"
                print(f"{tag} ...", end=" ", flush=True)
                try:
                    res = _spawn(kernel, r, c, "rust", nr, n_t)
                    cache[_rkey(kernel, n_t, r, c)] = res
                    save_cache(cache)
                    sk = cache.get(_skey(kernel, r, c), {}).get("time")
                    sp = f"  ({sk/res['time']:.1f}×)" if sk else ""
                    print(f"{res['time']*1000:.1f} ms{sp}")
                except Exception as e:
                    print(f"FAILED: {e}")
    return cache


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if "--worker" in sys.argv:
        _run_worker()
        return

    import argparse
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # kernel family flags
    p.add_argument("--l1",  action="store_true", help="include L1 normalize")
    p.add_argument("--l2",  action="store_true", help="include L2 normalize")
    p.add_argument("--max", action="store_true", help="include Max normalize")
    p.add_argument("--no-inplace", action="store_true", help="drop in-place kernel variants")

    # size tiers
    p.add_argument("--small",   action="store_true", help="10k–1M rows")
    p.add_argument("--large",   action="store_true", help="1M rows, wider")
    p.add_argument("--xlarge",  action="store_true", help="2M–5M rows")
    p.add_argument("--xxlarge", action="store_true", help="10M rows (~2 GB f32)")

    # threads / misc
    p.add_argument("--threads", default=None, metavar="LIST",
                   help=f"comma-separated thread counts (default {ALL_THREADS}, max {_n_cpu})")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the rust-vs-sklearn correctness check")
    p.add_argument("--plot-only", action="store_true", help="regenerate charts only")
    p.add_argument("--refresh",   action="store_true", help="ignore cache, re-run all")
    p.add_argument("--reps", type=int, default=N_REPS,
                   help=f"timing reps (default {N_REPS}, auto-reduced for large inputs)")
    args = p.parse_args()

    # kernel selection
    fam_selected = [f for f, on in [("l1", args.l1), ("l2", args.l2), ("max", args.max)] if on]
    if not fam_selected:
        fam_selected = FAMILIES
    kernels = [k for k, fam, _, ip in KERNELS
               if fam in fam_selected and not (args.no_inplace and ip)]

    # size selection (all tiers by default)
    if not (args.small or args.large or args.xlarge or args.xxlarge):
        args.small = args.large = args.xlarge = args.xxlarge = True
    seen, sizes = set(), []
    for tier in [SIZE_SMALL if args.small else [], SIZE_LARGE if args.large else [],
                 SIZE_XLARGE if args.xlarge else [], SIZE_XXLARGE if args.xxlarge else []]:
        for sz in tier:
            if sz not in seen:
                seen.add(sz); sizes.append(sz)

    # thread selection
    if args.threads:
        raw = [int(t) for t in args.threads.split(",")]
        thread_counts = sorted(set(t for t in raw if 1 <= t <= _n_cpu))
        ignored = [t for t in raw if t > _n_cpu]
        if ignored:
            print(f"Warning: {ignored} exceed CPU count ({_n_cpu}) — ignored.")
    else:
        thread_counts = ALL_THREADS
    if not thread_counts:
        print(f"No valid thread counts (max {_n_cpu}).")
        return

    print(f"Kernels  : {len(kernels)}  ({', '.join(fam_selected)})")
    print(f"Sizes    : {len(sizes)} datasets")
    print(f"Threads  : {thread_counts}  (machine has {_n_cpu} cores)")
    print()

    cache = {} if args.refresh else load_cache()
    if not args.plot_only:
        cache = run_benchmarks(cache, kernels, sizes, thread_counts, args.reps,
                               no_verify=args.no_verify)

    # plot only what has cached data
    plot_sizes = [s for s in sizes if any(_skey(k, *s) in cache for k in kernels)]
    plot_kernels = [k for k in kernels
                    if any(_skey(k, *s) in cache for s in plot_sizes)]
    plot_threads = [t for t in thread_counts
                    if any(_rkey(k, t, *s) in cache for k in plot_kernels for s in plot_sizes)]
    if not plot_sizes or not plot_kernels:
        print("\nNo cached results to plot yet. Run without --plot-only first.")
        return
    if not plot_threads:
        plot_threads = thread_counts

    print(f"\n── Generating charts ({len(plot_kernels)} kernels × "
          f"{len(plot_sizes)} sizes × {len(plot_threads)} threads) ──")
    plot_groupA(cache, plot_kernels, plot_sizes, plot_threads, "speedup")
    plot_groupA(cache, plot_kernels, plot_sizes, plot_threads, "runtime")
    plot_groupB(cache, plot_kernels, plot_sizes, plot_threads, "speedup")
    plot_groupB(cache, plot_kernels, plot_sizes, plot_threads, "runtime")
    print("\nDone.")


if __name__ == "__main__":
    main()
