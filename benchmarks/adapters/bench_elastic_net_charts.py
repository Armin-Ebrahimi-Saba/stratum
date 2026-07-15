#!/usr/bin/env python3
"""
ElasticNet staged benchmark — combinatorial speedup and runtime charts.

Group A  compare all commits vs sklearn across every (size, thread-count) combination.
         Produces two PDFs:  elastic_net_groupA_speedup.pdf  /  elastic_net_groupA_runtime.pdf
         Layout: rows = selected thread counts, cols = fit | predict.
         Lines per subplot = 3 commits + sklearn baseline.

Group B  HEAD vs sklearn — thread-count scaling for every selected input size.
         Produces two PDFs:  elastic_net_groupB_speedup.pdf  /  elastic_net_groupB_runtime.pdf
         Layout: 1 row × 2 cols (fit | predict).
         Lines = one per selected size, X = thread counts.

Execution is fully stageable — data is collected and saved cell-by-cell, so
a partial run can be resumed without re-running completed work.

  Size flags (independent — combine freely):
    --small    10k-100k rows  (4 sizes)
    --large    500k-1M rows   (2 sizes)
    --xlarge   2M-5M rows     (2 sizes)
    --xxlarge  10M rows       (1 size, ~2 GB f32)
  If no size flag is given, all tiers are selected.

  Thread flags:
    --threads 1,2,4,8        comma-separated subset (default: 1 through N_CPU)
    Maximum is fetched from os.cpu_count() — never hardcoded.

Other flags:
    --no-verify              skip correctness checks
    --no-sklearn-xlarge      skip sklearn for xlarge sizes (saves RAM)
    --plot-only              regenerate charts without re-running benchmarks
    --refresh                ignore cache, re-run all
    --reps N                 timing repetitions (default 5, auto-reduced for large inputs)

Shares the cache with bench_elastic_net_full.py
  (benchmarks/results/elastic_net_full_cache.json)

Examples:
    uv run python benchmarks/adapters/bench_elastic_net_charts.py --small
    uv run python benchmarks/adapters/bench_elastic_net_charts.py --large --xlarge
    uv run python benchmarks/adapters/bench_elastic_net_charts.py --small --threads 1,4
    uv run python benchmarks/adapters/bench_elastic_net_charts.py --plot-only
    uv run python benchmarks/adapters/bench_elastic_net_charts.py --small --no-verify
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
RESULTS    = REPO_ROOT / "benchmarks" / "results"
CACHE_FILE = RESULTS / "elastic_net_full_cache.json"

# ── commits (oldest → newest) ─────────────────────────────────────────────────
COMMITS = [
    {"hash": "cdeb7a3", "label": "Jacobi two-pass"},
    {"hash": "0ad4263", "label": "Fused single-pass"},
    {"hash": "10f16bd", "label": "Block-parallel"},
]
HEAD_COMMIT = COMMITS[-1]["hash"]

# ── thread counts — N_CPU is fetched dynamically, never hardcoded ─────────────
_n_cpu      = os.cpu_count() or 4
ALL_THREADS = sorted(set(t for t in [1, 2, 4, 8, _n_cpu] if t <= _n_cpu))

# ── size tiers ────────────────────────────────────────────────────────────────
SIZE_SMALL   = [(10_000, 50), (50_000, 50), (100_000, 50), (100_000, 100)]
SIZE_LARGE   = [(500_000, 50), (1_000_000, 20)]
SIZE_XLARGE  = [(2_000_000, 50), (5_000_000, 20)]
SIZE_XXLARGE = [(10_000_000, 50)]

# ── benchmark config ──────────────────────────────────────────────────────────
ALPHA    = 0.1
L1_RATIO = 0.5
MAX_ITER = 1000
TOL      = 1e-4
N_REPS   = 5

VERIFY_MAX_ITER = 10_000
VERIFY_TOL      = 1e-6
VERIFY_RTOL     = 0.05

# ── visual style ──────────────────────────────────────────────────────────────
COMMIT_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]   # blue, orange, green
SKLEARN_COLOR = "#9E9E9E"
THREAD_STYLES = {1: "-", 2: "--", 4: ":", 8: "-."}


# ─── workers (each runs inside a fresh subprocess so OnceLock inits correctly) ──

def _make_data(n_rows: int, n_cols: int, dtype):
    import numpy as np
    rng  = np.random.default_rng(42)
    X    = rng.standard_normal((n_rows, n_cols)).astype(dtype)
    coef = rng.standard_normal(n_cols).astype(dtype)
    y    = (X @ coef + 0.1 * rng.standard_normal(n_rows)).astype(dtype)
    return X, y


def _timeit(fn, n_reps: int) -> float:
    import numpy as np
    times = []
    for _ in range(n_reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _worker_rust(n_rows: int, n_cols: int, n_reps: int):
    import numpy as np
    from stratum import _rust_backend as rb
    X, y = _make_data(n_rows, n_cols, np.float32)
    for _ in range(2):
        rb.elastic_net_fit(X, y, ALPHA, L1_RATIO, MAX_ITER, TOL, True)
    gc.collect()
    fit_t = _timeit(lambda: rb.elastic_net_fit(X, y, ALPHA, L1_RATIO, MAX_ITER, TOL, True), n_reps)
    mid, *_ = rb.elastic_net_fit(X, y, ALPHA, L1_RATIO, MAX_ITER, TOL, True)
    for _ in range(2):
        rb.elastic_net_predict(mid, X)
    gc.collect()
    pred_t = _timeit(lambda: rb.elastic_net_predict(mid, X), n_reps)
    print(json.dumps({"fit": fit_t, "predict": pred_t}))


def _worker_sklearn(n_rows: int, n_cols: int, n_reps: int):
    import numpy as np
    from sklearn.linear_model import ElasticNet
    X, y = _make_data(n_rows, n_cols, np.float64)
    kw = dict(alpha=ALPHA, l1_ratio=L1_RATIO, max_iter=MAX_ITER, tol=TOL, fit_intercept=True)
    for _ in range(2):
        ElasticNet(**kw).fit(X, y)
    gc.collect()
    fit_t = _timeit(lambda: ElasticNet(**kw).fit(X, y), n_reps)
    sk = ElasticNet(**kw).fit(X, y)
    for _ in range(2):
        sk.predict(X)
    gc.collect()
    pred_t = _timeit(lambda: sk.predict(X), n_reps)
    print(json.dumps({"fit": fit_t, "predict": pred_t}))


def _worker_verify(n_rows: int, n_cols: int):
    import numpy as np
    from sklearn.linear_model import ElasticNet
    from stratum import _rust_backend as rb
    X_f32, y_f32 = _make_data(n_rows, n_cols, np.float32)
    X_f64, y_f64 = X_f32.astype(np.float64), y_f32.astype(np.float64)
    kw = dict(alpha=ALPHA, l1_ratio=L1_RATIO,
              max_iter=VERIFY_MAX_ITER, tol=VERIFY_TOL, fit_intercept=True)
    sk = ElasticNet(**kw).fit(X_f64, y_f64)
    sk_preds = sk.predict(X_f64).astype(np.float32)
    mid, coef_ru, *_ = rb.elastic_net_fit(
        X_f32, y_f32, ALPHA, L1_RATIO, VERIFY_MAX_ITER, VERIFY_TOL, True)
    ru_preds = rb.elastic_net_predict(mid, X_f32)
    rel_err = np.abs(ru_preds - sk_preds) / (np.abs(sk_preds) + 1e-6)
    print(json.dumps({
        "passed":           bool(rel_err.max() <= VERIFY_RTOL),
        "max_pred_rel_err": float(rel_err.max()),
        "coef_mse":         float(np.mean((coef_ru - sk.coef_.astype(np.float32)) ** 2)),
    }))


def _run_worker():
    a    = sys.argv
    mode = a[a.index("--mode") + 1]
    r    = int(a[a.index("--n-rows") + 1])
    c    = int(a[a.index("--n-cols") + 1])
    if mode == "verify":
        _worker_verify(r, c)
    elif mode == "sklearn":
        _worker_sklearn(r, c, int(a[a.index("--n-reps") + 1]))
    else:
        _worker_rust(r, c, int(a[a.index("--n-reps") + 1]))


# ─── subprocess launcher ───────────────────────────────────────────────────────

def _spawn(n_rows: int, n_cols: int, mode: str,
           n_reps: int = N_REPS, n_threads: int | None = None) -> dict:
    env = dict(os.environ)
    env["SKRUB_RUST"] = "1"
    if n_threads is not None:
        env["SKRUB_RUST_THREADS"] = str(n_threads)
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker", "--mode", mode,
           "--n-rows", str(n_rows), "--n-cols", str(n_cols)]
    if mode != "verify":
        cmd += ["--n-reps", str(n_reps)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO_ROOT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-800:])
    return json.loads(r.stdout.strip())


# ─── git / build ──────────────────────────────────────────────────────────────

def _create_worktree(commit_hash: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix=f"enet-charts-{commit_hash[:7]}-"))
    r = subprocess.run(
        ["git", "worktree", "add", "--detach", str(tmp), commit_hash],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(r.stderr.strip())
    return tmp


def _remove_worktree(wt: Path):
    subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                   cwd=REPO_ROOT, capture_output=True, text=True)
    shutil.rmtree(wt, ignore_errors=True)


def _build(rust_dir: Path) -> bool:
    venv_mat = REPO_ROOT / ".venv" / "bin" / "maturin"
    candidates = (
        [[str(venv_mat), "develop", "--release"]] if venv_mat.exists() else []
    ) + [
        ["maturin", "develop", "--release"],
        ["uv", "run", "maturin", "develop", "--release"],
    ]
    for cmd in candidates:
        r = subprocess.run(cmd, cwd=rust_dir, capture_output=True, text=True)
        if r.returncode == 0:
            return True
    print(f"  build FAILED:\n{r.stderr[-400:]}")
    return False


# ─── cache ────────────────────────────────────────────────────────────────────

def _tkey(commit: str, n_threads: int, n_rows: int, n_cols: int) -> str:
    return f"{commit}__t{n_threads}__{n_rows}__{n_cols}"

def _skey(n_rows: int, n_cols: int) -> str:
    return f"sklearn__{n_rows}__{n_cols}"

def _vkey(commit: str, n_rows: int, n_cols: int) -> str:
    return f"verify__{commit}__{n_rows}__{n_cols}"

def load_cache() -> dict:
    return json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}

def save_cache(cache: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def reps_for(n_rows: int, n_cols: int, base: int = N_REPS) -> int:
    mb = n_rows * n_cols * 4 / 1e6
    if mb > 500: return min(base, 3)
    if mb > 100: return min(base, 4)
    return base


# ─── plotting ─────────────────────────────────────────────────────────────────

def _size_label(n_rows: int, n_cols: int) -> str:
    mb = n_rows * n_cols * 4 / 1e6
    return f"{n_rows // 1000}k×{n_cols}\n({mb:.0f} MB)"


def _ms_fmt(v, _):
    return f"{v:.0f}" if v >= 1 else f"{v:.2f}"


def _sx_fmt(v, _):
    return f"{v:.0f}×" if v >= 2 else f"{v:.1f}×"


def plot_groupA(cache: dict, sizes: list, thread_counts: list, metric: str):
    """
    Group A — all commits × thread counts × input sizes.
    Layout: rows = thread counts, cols = fit | predict.
    Lines per subplot: one per commit + sklearn reference.
    metric: 'speedup'  →  Y = speedup over sklearn (log scale)
            'runtime'  →  Y = median time in ms (log scale)
    Output: elastic_net_groupA_{metric}.pdf
    """
    import matplotlib; matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5,
                         "legend.fontsize": 7.5, "xtick.labelsize": 7.5})

    xs     = list(range(len(sizes)))
    labels = [_size_label(r, c) for r, c in sizes]
    nrows  = len(thread_counts)

    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.2 * nrows), squeeze=False)
    title = ("Speedup over sklearn" if metric == "speedup" else "Runtime (ms, log scale)")
    fig.suptitle(f"Group A — All commits vs sklearn — {title}  [{_n_cpu} CPU cores]",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, n_t in enumerate(thread_counts):
        ls = THREAD_STYLES.get(n_t, "-")
        for col, phase in enumerate(("fit", "predict")):
            ax = axes[row][col]
            sk_vals = [cache.get(_skey(r, c), {}).get(phase) for r, c in sizes]

            if metric == "speedup":
                ax.axhline(1.0, color=SKLEARN_COLOR, lw=1.2, ls="--",
                           label="sklearn (1×)", zorder=1)
            else:
                sk_vx = [j for j, v in enumerate(sk_vals) if v is not None]
                sk_vy = [sk_vals[j] * 1000 for j in sk_vx]
                if sk_vx:
                    ax.plot(sk_vx, sk_vy, color=SKLEARN_COLOR, lw=2, ls="--",
                            marker="s", ms=4, label="sklearn", zorder=1)

            for ci, commit in enumerate(COMMITS):
                h     = commit["hash"]
                times = [cache.get(_tkey(h, n_t, r, c), {}).get(phase) for r, c in sizes]
                vx, vy = [], []
                for j, (t, s) in enumerate(zip(times, sk_vals)):
                    if t is not None and t > 0:
                        if metric == "speedup" and s:
                            vx.append(j); vy.append(s / t)
                        elif metric == "runtime":
                            vx.append(j); vy.append(t * 1000)
                if not vy:
                    continue
                ax.plot(vx, vy, color=COMMIT_COLORS[ci], lw=1.8, ls=ls,
                        marker="o", ms=5, label=commit["label"], zorder=2, alpha=0.9)

            ax.set_yscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Dataset size")
            if metric == "speedup":
                ax.set_ylabel("Speedup (×)")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(_sx_fmt))
            else:
                ax.set_ylabel("Median time (ms)")
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(_ms_fmt))
            phase_lbl = "Fit" if phase == "fit" else "Predict"
            thread_lbl = f"{n_t} thread{'s' if n_t > 1 else ''}"
            ax.set_title(f"{phase_lbl}  —  {thread_lbl}", pad=5)
            ax.legend(fontsize=7.5, loc="upper left")
            ax.grid(True, alpha=0.2, which="both")
            ax.set_xlim(-0.4, len(sizes) - 0.6)

    fig.tight_layout()
    out = RESULTS / f"elastic_net_groupA_{metric}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def plot_groupB(cache: dict, sizes: list, thread_counts: list, metric: str):
    """
    Group B — HEAD vs sklearn, thread-count scaling for all selected sizes.
    Layout: 1 row × 2 cols (fit | predict).
    Lines: one per selected size, X = thread counts.
    metric: 'speedup'  →  Y = speedup over sklearn (log scale)
            'runtime'  →  Y = HEAD median time in ms (log scale)
    Output: elastic_net_groupB_{metric}.pdf
    """
    import matplotlib; matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    size_colors = [plt.cm.tab10(i) for i in range(len(sizes))]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "legend.fontsize": 8, "xtick.labelsize": 9})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    title = ("Speedup over sklearn" if metric == "speedup" else "Runtime (ms, log scale)")
    fig.suptitle(
        f"Block-parallel predict vs sklearn — {title}  [{_n_cpu} CPU cores]",
        fontsize=13, fontweight="bold", y=1.01)

    for col, phase in enumerate(("fit", "predict")):
        ax = axes[col]

        if metric == "speedup":
            ax.axhline(1.0, color=SKLEARN_COLOR, lw=1.2, ls="--",
                       label="sklearn (1×)", zorder=1)

        for si, (n_rows, n_cols) in enumerate(sizes):
            sk_t = cache.get(_skey(n_rows, n_cols), {}).get(phase)
            tx, ty = [], []
            for n_t in thread_counts:
                ru_t = cache.get(_tkey(HEAD_COMMIT, n_t, n_rows, n_cols), {}).get(phase)
                if ru_t and ru_t > 0:
                    tx.append(n_t)
                    if metric == "speedup" and sk_t:
                        ty.append(sk_t / ru_t)
                    elif metric == "runtime":
                        ty.append(ru_t * 1000)
            if not ty:
                continue
            mb  = n_rows * n_cols * 4 / 1e6
            lbl = f"{n_rows // 1000}k×{n_cols} ({mb:.0f} MB)"
            ax.plot(tx, ty, color=size_colors[si], lw=2, marker="o", ms=5,
                    label=lbl, zorder=2)

            if metric == "runtime" and sk_t:
                ax.axhline(sk_t * 1000, color=size_colors[si],
                           lw=1, ls="--", alpha=0.4, zorder=1)

        ax.set_yscale("log")
        ax.set_xticks(thread_counts)
        ax.set_xlabel("Thread count")
        if metric == "speedup":
            ax.set_ylabel("Speedup over sklearn (×)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_sx_fmt))
        else:
            ax.set_ylabel("Median time (ms)  [dashed = sklearn]")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_ms_fmt))
        ax.set_title("Fit" if phase == "fit" else "Predict", pad=5)
        ax.legend(fontsize=7.5, loc="upper right" if metric == "speedup" else "lower left",
                  ncol=1 + len(sizes) // 6)
        ax.grid(True, alpha=0.2, which="both")

    fig.tight_layout()
    out = RESULTS / f"elastic_net_groupB_{metric}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


# ─── benchmark runner ─────────────────────────────────────────────────────────

def run_benchmarks(cache: dict, all_sizes: list, thread_counts: list,
                   args) -> dict:
    """Collect all missing cells; saves cache after every cell."""

    # ── sklearn baseline ──────────────────────────────────────────────────────
    sk_missing = [(r, c) for r, c in all_sizes if _skey(r, c) not in cache]
    if sk_missing:
        print("── sklearn baseline ──")
    for n_rows, n_cols in sk_missing:
        mb = n_rows * n_cols * 4 / 1e6
        if args.no_sklearn_xlarge and mb > 200:
            print(f"  {n_rows:>8,}×{n_cols} ({mb:.0f} MB)  skipped (--no-sklearn-xlarge)")
            continue
        print(f"  {n_rows:>8,}×{n_cols} ({mb:.0f} MB) ...", end=" ", flush=True)
        try:
            res = _spawn(n_rows, n_cols, "sklearn", reps_for(n_rows, n_cols, args.reps))
            cache[_skey(n_rows, n_cols)] = res
            save_cache(cache)
            print(f"fit={res['fit']*1000:.1f}ms  predict={res['predict']*1000:.1f}ms")
        except Exception as e:
            print(f"FAILED: {e}")

    # ── per-commit benchmarks ─────────────────────────────────────────────────
    for commit in COMMITS:
        h     = commit["hash"]
        label = commit["label"]

        missing_timing = [
            (n_t, r, c)
            for n_t in thread_counts
            for r, c in all_sizes
            if _tkey(h, n_t, r, c) not in cache
        ]
        missing_verify = [] if args.no_verify else [
            (r, c) for r, c in SIZE_SMALL
            if _vkey(h, r, c) not in cache
        ]

        if not missing_timing and not missing_verify:
            print(f"\n── {h} ({label}): all cells cached ──")
            continue

        print(f"\n── {h} ({label}) ──")
        print(f"   {len(missing_timing)} timing cell(s) + {len(missing_verify)} verify cell(s)")
        print("   building ...", end=" ", flush=True)
        worktree = None
        try:
            worktree = _create_worktree(h)
            if not _build(worktree / "_rust"):
                print("FAILED — skipping this commit")
                continue
            print("done")

            if missing_verify:
                print(f"   correctness (rtol={VERIFY_RTOL}) ──")
                any_fail = False
                for n_rows, n_cols in missing_verify:
                    mb  = n_rows * n_cols * 4 / 1e6
                    tag = f"  {n_rows:>8,}×{n_cols} ({mb:.0f} MB)"
                    print(f"{tag} ...", end=" ", flush=True)
                    try:
                        v = _spawn(n_rows, n_cols, "verify", n_threads=1)
                        cache[_vkey(h, n_rows, n_cols)] = v
                        save_cache(cache)
                        st = "PASS ✓" if v["passed"] else "FAIL ✗"
                        print(f"{st}  max_rel={v['max_pred_rel_err']:.2e}"
                              f"  coef_mse={v['coef_mse']:.2e}")
                        if not v["passed"]:
                            any_fail = True
                    except Exception as e:
                        print(f"ERROR: {e}"); any_fail = True
                if any_fail:
                    print("   Correctness failed — aborting this commit.")
                    continue

            print("   timing ──")
            for n_t, n_rows, n_cols in missing_timing:
                mb  = n_rows * n_cols * 4 / 1e6
                k   = _tkey(h, n_t, n_rows, n_cols)
                tag = f"  {n_rows:>8,}×{n_cols} ({mb:.0f} MB)  {n_t}T"
                print(f"{tag} ...", end=" ", flush=True)
                try:
                    res = _spawn(n_rows, n_cols, "rust",
                                 reps_for(n_rows, n_cols, args.reps), n_t)
                    cache[k] = res
                    save_cache(cache)
                    sk_fit = cache.get(_skey(n_rows, n_cols), {}).get("fit")
                    sp = f"  ({sk_fit/res['fit']:.1f}× fit)" if sk_fit else ""
                    print(f"fit={res['fit']*1000:.1f}ms"
                          f"  predict={res['predict']*1000:.1f}ms{sp}")
                except Exception as e:
                    print(f"FAILED: {e}")

        finally:
            if worktree:
                _remove_worktree(worktree)

    # ── rebuild HEAD so the active extension is the current version ───────────
    print("\n── Rebuilding HEAD extension ──")
    ok = _build(REPO_ROOT / "_rust")
    print("done" if ok else "FAILED — run: cd _rust && maturin develop --release")

    return cache


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if "--worker" in sys.argv:
        _run_worker()
        return

    import argparse
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # size tier flags — INDEPENDENT, not cumulative
    p.add_argument("--small",   action="store_true",
                   help="include small sizes   (10k–100k rows)")
    p.add_argument("--large",   action="store_true",
                   help="include large sizes   (500k–1M rows)")
    p.add_argument("--xlarge",  action="store_true",
                   help="include xlarge sizes  (2M–5M rows)")
    p.add_argument("--xxlarge", action="store_true",
                   help="include xxlarge sizes (10M×50, ~2 GB f32)")

    # thread selection
    p.add_argument("--threads", default=None, metavar="LIST",
                   help=f"comma-separated thread counts to run/plot "
                        f"(default: {ALL_THREADS}, max = {_n_cpu} cores)")

    # misc
    p.add_argument("--no-verify", action="store_true",
                   help="skip correctness checks")
    p.add_argument("--no-sklearn-xlarge", action="store_true",
                   help="skip sklearn for xlarge sizes (saves RAM)")
    p.add_argument("--plot-only", action="store_true",
                   help="regenerate charts without re-running benchmarks")
    p.add_argument("--refresh", action="store_true",
                   help="ignore cache and re-run everything")
    p.add_argument("--reps", type=int, default=N_REPS,
                   help=f"timing repetitions (default {N_REPS}, "
                        "auto-reduced for large inputs)")
    args = p.parse_args()

    # default: all tiers when no size flag is given
    if not (args.small or args.large or args.xlarge or args.xxlarge):
        args.small = args.large = args.xlarge = args.xxlarge = True

    # parse and validate thread counts
    if args.threads:
        raw = [int(t.strip()) for t in args.threads.split(",")]
        thread_counts = sorted(set(t for t in raw if 1 <= t <= _n_cpu))
        ignored = [t for t in raw if t > _n_cpu]
        if ignored:
            print(f"Warning: thread counts {ignored} exceed CPU count "
                  f"({_n_cpu}) and will be ignored.")
    else:
        thread_counts = ALL_THREADS

    if not thread_counts:
        print(f"No valid thread counts after filtering (max = {_n_cpu} cores).")
        return

    # build ordered size list (preserve tier order, deduplicate)
    seen: set = set()
    all_sizes: list = []
    for tier_sizes in [
        SIZE_SMALL   if args.small   else [],
        SIZE_LARGE   if args.large   else [],
        SIZE_XLARGE  if args.xlarge  else [],
        SIZE_XXLARGE if args.xxlarge else [],
    ]:
        for sz in tier_sizes:
            if sz not in seen:
                seen.add(sz); all_sizes.append(sz)

    print(f"Tiers    : {'small ' if args.small else ''}"
          f"{'large ' if args.large else ''}"
          f"{'xlarge ' if args.xlarge else ''}"
          f"{'xxlarge' if args.xxlarge else ''}")
    print(f"Threads  : {thread_counts}  (machine has {_n_cpu} cores)")
    print(f"Sizes    : {len(all_sizes)} datasets")
    print(f"Commits  : {len(COMMITS)}")
    print()

    cache = {} if args.refresh else load_cache()

    if not args.plot_only:
        cache = run_benchmarks(cache, all_sizes, thread_counts, args)

    # restrict plot to sizes and thread counts that have actual cached data
    plot_sizes = [
        sz for sz in all_sizes
        if _skey(*sz) in cache
    ]
    plot_threads = [
        n_t for n_t in thread_counts
        if any(
            _tkey(cm["hash"], n_t, r, c) in cache
            for cm in COMMITS for r, c in plot_sizes
        )
    ]

    if not plot_sizes:
        print("\nNo cached results to plot yet. Run without --plot-only first.")
        return

    if not plot_threads:
        # fall back to requested counts even if no data — plots will just be empty
        plot_threads = thread_counts

    print(f"\n── Generating charts ({len(plot_sizes)} sizes × "
          f"{len(plot_threads)} thread counts) ──")

    print("Group A — all commits vs sklearn:")
    plot_groupA(cache, plot_sizes, plot_threads, "speedup")
    plot_groupA(cache, plot_sizes, plot_threads, "runtime")

    print("Group B — HEAD vs sklearn (thread scaling):")
    plot_groupB(cache, plot_sizes, plot_threads, "speedup")
    plot_groupB(cache, plot_sizes, plot_threads, "runtime")

    print("\nDone.")


if __name__ == "__main__":
    main()
