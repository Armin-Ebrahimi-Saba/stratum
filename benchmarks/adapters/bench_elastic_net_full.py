#!/usr/bin/env python3
"""
Full ElasticNet benchmark — commits × thread counts × dataset sizes.

Single cache file. All results reproducible by re-running with the same flags.

Generates three plots:
  benchmarks/results/elastic_net_speedup.pdf       — speedup over sklearn
  benchmarks/results/elastic_net_runtime.pdf       — absolute runtime in ms
  benchmarks/results/elastic_net_thread_scaling.pdf — thread-count scaling per commit

Usage (from repo root):
    uv run python benchmarks/adapters/bench_elastic_net_full.py
    uv run python benchmarks/adapters/bench_elastic_net_full.py --large
    uv run python benchmarks/adapters/bench_elastic_net_full.py --xlarge
    uv run python benchmarks/adapters/bench_elastic_net_full.py --no-verify
    uv run python benchmarks/adapters/bench_elastic_net_full.py --no-sklearn-xlarge
    uv run python benchmarks/adapters/bench_elastic_net_full.py --plot-only
    uv run python benchmarks/adapters/bench_elastic_net_full.py --refresh

Size flags are cumulative: --xlarge implies --large.
Builds each commit once per run; unmodified cached cells are skipped.
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

# ── paths ───────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
RESULTS    = REPO_ROOT / "benchmarks" / "results"
CACHE_FILE = RESULTS / "elastic_net_full_cache.json"

# ── commits (oldest → newest) ────────────────────────────────────────────────────
COMMITS = [
    {"hash": "cdeb7a3", "label": "Jacobi two-pass"},
    {"hash": "0ad4263", "label": "Fused single-pass"},
    {"hash": "10f16bd", "label": "Block-parallel predict (HEAD)"},
]

# ── thread counts (filtered to available cores) ──────────────────────────────────
_n_cpu        = os.cpu_count() or 4
THREAD_COUNTS = sorted(set(t for t in [1, 2, 4, 8, _n_cpu] if t <= _n_cpu))

# ── size tiers ───────────────────────────────────────────────────────────────────
SIZE_DEFAULT = [(10_000, 50), (50_000, 50), (100_000, 50), (100_000, 100)]
SIZE_LARGE   = [(500_000, 50), (1_000_000, 20)]
SIZE_XLARGE  = [(2_000_000, 50), (5_000_000, 20)]

# ── benchmark config ─────────────────────────────────────────────────────────────
ALPHA    = 0.1
L1_RATIO = 0.5
MAX_ITER = 1000
TOL      = 1e-4
N_REPS   = 5

VERIFY_MAX_ITER = 10_000
VERIFY_TOL      = 1e-6
VERIFY_RTOL     = 0.05


# ─── workers (each runs inside a subprocess after the right .so is installed) ────

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
        "passed":            bool(rel_err.max() <= VERIFY_RTOL),
        "max_pred_rel_err":  float(rel_err.max()),
        "coef_mse":          float(np.mean((coef_ru - sk.coef_.astype(np.float32)) ** 2)),
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


# ─── subprocess launcher ─────────────────────────────────────────────────────────

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


# ─── git / build ─────────────────────────────────────────────────────────────────

def _create_worktree(commit_hash: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix=f"enet-bench-{commit_hash[:7]}-"))
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
    for cmd in (
        [[str(venv_mat), "develop", "--release"]] if venv_mat.exists() else []
    ) + [["maturin", "develop", "--release"],
         ["uv", "run", "maturin", "develop", "--release"]]:
        r = subprocess.run(cmd, cwd=rust_dir, capture_output=True, text=True)
        if r.returncode == 0:
            return True
    print(f"  build FAILED:\n{r.stderr[-400:]}")
    return False


# ─── cache helpers ────────────────────────────────────────────────────────────────

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


# ─── plotting ─────────────────────────────────────────────────────────────────────

COMMIT_COLORS  = ["#2196F3", "#FF9800", "#4CAF50"]   # blue, orange, green
SKLEARN_COLOR  = "#9E9E9E"
THREAD_STYLES  = {1: ("-",  "o"), 2: ("--", "s"), 4: (":",  "^"), 8: ("-.", "D")}


def _size_labels(sizes):
    return [
        f"{r//1000}k×{c}\n({r*c*4/1e6:.0f} MB)" for r, c in sizes
    ]


def _get_times(cache, commit, n_threads, sizes, phase):
    return [cache.get(_tkey(commit, n_threads, r, c), {}).get(phase) for r, c in sizes]


def _get_sklearn(cache, sizes, phase):
    return [cache.get(_skey(r, c), {}).get(phase) for r, c in sizes]


def plot_speedup(cache: dict, sizes: list):
    """1-row × 2-col: speedup over sklearn, lines = commit × thread."""
    import matplotlib; matplotlib.use("pdf")
    import matplotlib.pyplot as plt, matplotlib.ticker as mticker

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "legend.fontsize": 8, "xtick.labelsize": 7.5})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("ElasticNet speedup over sklearn — all commits and thread counts",
                 fontsize=13, fontweight="bold", y=1.01)

    xs     = list(range(len(sizes)))
    labels = _size_labels(sizes)

    for phase, ax in zip(("fit", "predict"), axes):
        sk = _get_sklearn(cache, sizes, phase)
        ax.axhline(1.0, color=SKLEARN_COLOR, linewidth=1.2, linestyle="--",
                   label="sklearn (1×)", zorder=1)

        for ci, commit in enumerate(COMMITS):
            h = commit["hash"]
            for n_t in THREAD_COUNTS:
                ls, mk = THREAD_STYLES.get(n_t, ("-", "o"))
                times  = _get_times(cache, h, n_t, sizes, phase)
                vx, vy = [], []
                for j, (t, s) in enumerate(zip(times, sk)):
                    if t and s and t > 0:
                        vx.append(xs[j]); vy.append(s / t)
                if not vy:
                    continue
                lbl = f"{commit['label']} {n_t}T"
                ax.plot(vx, vy, color=COMMIT_COLORS[ci], linestyle=ls,
                        marker=mk, markersize=5, linewidth=1.8,
                        label=lbl, alpha=0.9, zorder=2)

        ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("Dataset size"); ax.set_ylabel("Speedup (×)")
        ax.set_title(f"{'Fit' if phase=='fit' else 'Predict'}", pad=6)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.2, which="both")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.0f}×" if v >= 2 else f"{v:.1f}×"))
        ax.set_xlim(-0.4, len(sizes) - 0.6)

    fig.tight_layout()
    out = RESULTS / "elastic_net_speedup.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def plot_runtime(cache: dict, sizes: list):
    """2-row × 2-col: absolute runtime; rows = 1T and 4T, cols = fit/predict."""
    import matplotlib; matplotlib.use("pdf")
    import matplotlib.pyplot as plt, matplotlib.ticker as mticker

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "legend.fontsize": 8, "xtick.labelsize": 7.5})

    show_threads = [t for t in [1, 4] if t in THREAD_COUNTS]
    fig, axes = plt.subplots(len(show_threads), 2,
                             figsize=(14, 4.5 * len(show_threads)), sharey=False)
    if len(show_threads) == 1:
        axes = [axes]

    fig.suptitle("ElasticNet runtime — each commit at 1T and 4T vs sklearn",
                 fontsize=13, fontweight="bold", y=1.01)

    xs = list(range(len(sizes)))
    labels = _size_labels(sizes)

    for row, n_t in enumerate(show_threads):
        for col, phase in enumerate(("fit", "predict")):
            ax = axes[row][col]

            # sklearn reference
            sk = _get_sklearn(cache, sizes, phase)
            vx = [j for j, t in enumerate(sk) if t]
            vy = [sk[j] * 1000 for j in vx]
            if vx:
                ax.plot(vx, vy, color=SKLEARN_COLOR, linestyle="--", linewidth=2,
                        marker="s", markersize=4, label="sklearn", zorder=1)

            for ci, commit in enumerate(COMMITS):
                times = _get_times(cache, commit["hash"], n_t, sizes, phase)
                vx2 = [j for j, t in enumerate(times) if t]
                vy2 = [times[j] * 1000 for j in vx2]
                if vx2:
                    ax.plot(vx2, vy2, color=COMMIT_COLORS[ci], linewidth=2,
                            marker="o", markersize=5, label=commit["label"], zorder=2)

            ax.set_yscale("log")
            ax.set_xticks(xs); ax.set_xticklabels(labels)
            ax.set_xlabel("Dataset size")
            ax.set_ylabel("Median time (ms)")
            ax.set_title(f"{'Fit' if phase=='fit' else 'Predict'}  —  {n_t}T", pad=5)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2, which="both")
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.2f}"))
            ax.set_xlim(-0.4, len(sizes) - 0.6)

    fig.tight_layout()
    out = RESULTS / "elastic_net_runtime.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def plot_thread_scaling(cache: dict, sizes: list):
    """3-row × 2-col: for each commit × phase, X = thread count, lines = sizes."""
    import matplotlib; matplotlib.use("pdf")
    import matplotlib.pyplot as plt, matplotlib.ticker as mticker

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5,
                         "legend.fontsize": 8, "xtick.labelsize": 8.5})

    size_colors = [plt.cm.tab10(i) for i in range(len(sizes))]
    mb_labels   = [f"{r//1000}k×{c} ({r*c*4/1e6:.0f} MB)" for r, c in sizes]

    fig, axes = plt.subplots(len(COMMITS), 2,
                             figsize=(13, 4.2 * len(COMMITS)), sharey=False)
    fig.suptitle("Thread-count scaling per commit — speedup over sklearn",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, commit in enumerate(COMMITS):
        h = commit["hash"]
        for col, phase in enumerate(("fit", "predict")):
            ax = axes[row][col]
            ax.axhline(1.0, color=SKLEARN_COLOR, linewidth=1, linestyle="--",
                       label="sklearn (1×)", zorder=1)

            for si, (n_rows, n_cols) in enumerate(sizes):
                sk_t = cache.get(_skey(n_rows, n_cols), {}).get(phase)
                if not sk_t:
                    continue
                tx, ty = [], []
                for n_t in THREAD_COUNTS:
                    ru_t = cache.get(_tkey(h, n_t, n_rows, n_cols), {}).get(phase)
                    if ru_t and ru_t > 0:
                        tx.append(n_t); ty.append(sk_t / ru_t)
                if ty:
                    ax.plot(tx, ty, color=size_colors[si], marker="o",
                            markersize=5, linewidth=1.8, label=mb_labels[si])

            ax.set_yscale("log")
            ax.set_xticks(THREAD_COUNTS)
            ax.set_xlabel("Thread count")
            ax.set_ylabel("Speedup over sklearn (×)")
            ax.set_title(f"{commit['label']}  —  {'Fit' if phase=='fit' else 'Predict'}", pad=5)
            if row == 0 and col == 1:
                ax.legend(fontsize=7.5, loc="upper left")
            ax.grid(True, alpha=0.2, which="both")
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:.0f}×" if v >= 2 else f"{v:.1f}×"))

    fig.tight_layout()
    out = RESULTS / "elastic_net_thread_scaling.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  → {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────────

def main():
    if "--worker" in sys.argv:
        _run_worker()
        return

    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--large",    action="store_true", help="add large sizes (500k, 1M rows)")
    p.add_argument("--xlarge",   action="store_true", help="add xlarge sizes (2M, 5M rows)")
    p.add_argument("--no-verify",action="store_true", help="skip correctness checks")
    p.add_argument("--no-sklearn-xlarge", action="store_true",
                   help="skip sklearn for xlarge sizes (saves RAM)")
    p.add_argument("--plot-only",action="store_true", help="regenerate plots only")
    p.add_argument("--refresh",  action="store_true", help="ignore cache, re-run everything")
    p.add_argument("--reps",     type=int, default=N_REPS,
                   help=f"timing repetitions (default {N_REPS}, auto-reduced for xlarge)")
    args = p.parse_args()

    # build size list
    sizes = list(SIZE_DEFAULT)
    if args.large or args.xlarge:
        sizes += SIZE_LARGE
    if args.xlarge:
        sizes += SIZE_XLARGE

    # reps: auto-reduce for large/xlarge
    def reps_for(n_rows, n_cols):
        mb = n_rows * n_cols * 4 / 1e6
        if mb > 500:  return min(args.reps, 3)
        if mb > 100:  return min(args.reps, 4)
        return args.reps

    cache = {} if args.refresh else load_cache()

    if not args.plot_only:
        # ── sklearn baseline ──────────────────────────────────────────────────
        sk_missing = [(r, c) for r, c in sizes if _skey(r, c) not in cache]
        if sk_missing:
            print("── sklearn baseline ──")
        for n_rows, n_cols in sk_missing:
            mb = n_rows * n_cols * 4 / 1e6
            if args.no_sklearn_xlarge and mb > 200:
                print(f"  {n_rows:>8,}×{n_cols}  ({mb:.0f} MB)  skipped (--no-sklearn-xlarge)")
                continue
            print(f"  {n_rows:>8,}×{n_cols}  ({mb:.0f} MB) ...", end=" ", flush=True)
            try:
                res = _spawn(n_rows, n_cols, "sklearn", reps_for(n_rows, n_cols))
                cache[_skey(n_rows, n_cols)] = res
                save_cache(cache)
                print(f"fit={res['fit']*1000:.1f}ms  predict={res['predict']*1000:.1f}ms")
            except Exception as e:
                print(f"FAILED: {e}")

        # ── per-commit benchmarks ─────────────────────────────────────────────
        for commit in COMMITS:
            h     = commit["hash"]
            label = commit["label"]

            # check which (thread, size) cells are missing
            missing_timing = [
                (n_t, r, c) for n_t in THREAD_COUNTS for r, c in sizes
                if _tkey(h, n_t, r, c) not in cache
            ]
            missing_verify = [
                (r, c) for r, c in SIZE_DEFAULT   # verify only on default sizes
                if not args.no_verify and _vkey(h, r, c) not in cache
            ]

            if not missing_timing and not missing_verify:
                print(f"\n── {h} ({label}): all cells cached ──")
                continue

            print(f"\n── {h} ({label}) ──")
            print(f"   {len(missing_timing)} timing cells + {len(missing_verify)} verify cells")
            print("   building ...", end=" ", flush=True)
            worktree = None
            try:
                worktree = _create_worktree(h)
                if not _build(worktree / "_rust"):
                    print("FAILED — skipping")
                    continue
                print("done")

                # correctness checks
                if missing_verify:
                    print(f"   correctness (rtol={VERIFY_RTOL}) ──")
                    any_fail = False
                    for n_rows, n_cols in missing_verify:
                        mb  = n_rows * n_cols * 4 / 1e6
                        tag = f"  {n_rows:>8,}×{n_cols}  ({mb:.0f} MB)"
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

                # timing
                print(f"   timing ──")
                for n_t, n_rows, n_cols in missing_timing:
                    mb  = n_rows * n_cols * 4 / 1e6
                    k   = _tkey(h, n_t, n_rows, n_cols)
                    tag = f"  {n_rows:>8,}×{n_cols}  ({mb:.0f} MB)  {n_t}T"
                    print(f"{tag} ...", end=" ", flush=True)
                    try:
                        res = _spawn(n_rows, n_cols, "rust", reps_for(n_rows, n_cols), n_t)
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

        # ── rebuild HEAD ──────────────────────────────────────────────────────
        print("\n── Rebuilding HEAD extension ──")
        ok = _build(REPO_ROOT / "_rust")
        print("done" if ok else "FAILED — run: cd _rust && maturin develop --release")

    # ── plots ─────────────────────────────────────────────────────────────────
    # Use whatever sizes we have in cache (union of all cached sizes)
    cached_sizes = sorted(set(
        (int(k.split("__")[1]), int(k.split("__")[2]))
        for k in cache if k.startswith("sklearn__")
    ))
    plot_sizes = [s for s in sizes if s in cached_sizes] if not args.plot_only else cached_sizes

    if not plot_sizes:
        print("No cached results to plot.")
        return

    print("\n── Generating plots ──")
    plot_speedup(cache, plot_sizes)
    plot_runtime(cache, plot_sizes)
    plot_thread_scaling(cache, plot_sizes)


if __name__ == "__main__":
    main()
