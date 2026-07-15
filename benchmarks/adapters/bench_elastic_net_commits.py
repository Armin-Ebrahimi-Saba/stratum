#!/usr/bin/env python3
"""
Benchmark the impact of each optimization commit on ElasticNet performance.

For each commit the script:
  1. Creates a temporary git worktree
  2. Rebuilds the Rust extension with maturin
  3. Runs fit + predict timing at all dataset sizes
  4. Removes the worktree

HEAD is rebuilt after all commits so the repo is left in a clean state.

Results cached in benchmarks/results/elastic_net_commits.json.
Plot saved  to benchmarks/results/elastic_net_commits_plot.pdf.

Usage (from repo root):
    uv run python benchmarks/adapters/bench_elastic_net_commits.py
    uv run python benchmarks/adapters/bench_elastic_net_commits.py --plot-only
    uv run python benchmarks/adapters/bench_elastic_net_commits.py --refresh
    uv run python benchmarks/adapters/bench_elastic_net_commits.py --reps 3
    uv run python benchmarks/adapters/bench_elastic_net_commits.py --large
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
CACHE_FILE = RESULTS / "elastic_net_commits.json"
PLOT_FILE  = RESULTS / "elastic_net_commits_plot.pdf"

# ── commits to benchmark (oldest → newest) ─────────────────────────────────────
COMMITS = [
    {
        "hash":  "cdeb7a3",
        "label": "Jacobi\ntwo-pass",
        "fit_changed":     True,
        "predict_changed": True,
        "description": (
            "Initial parallel implementation. "
            "Two separate Rayon passes per iteration: "
            "pass 1 accumulates $X^\\top r$, pass 2 updates residuals. "
            "Predict: one task per row (fine-grained), "
            "full matrix copy across FFI boundary."
        ),
    },
    {
        "hash":  "0ad4263",
        "label": "Fused\nsingle-pass",
        "fit_changed":     True,
        "predict_changed": False,
        "description": (
            "Replaces two-pass fit with a cache-fused single pass. "
            "Each Rayon task owns a 256\\,KB block of $X$: "
            "phase~1 updates $r$ (cold $\\to$ L2), "
            "phase~2 accumulates $X^\\top r$ (L2-hot). "
            "One DRAM read instead of two. "
            "Predict unchanged from commit~1."
        ),
    },
    {
        "hash":  "10f16bd",
        "label": "Block-parallel\npredict (HEAD)",
        "fit_changed":     False,
        "predict_changed": True,
        "description": (
            "Fit identical to commit~2. "
            "Predict: borrows $X$ as \\texttt{ArrayView2} (zero FFI copy), "
            "rows batched into 256\\,KB blocks so each Rayon task "
            "does enough work to amortise scheduling overhead."
        ),
    },
]

# ── benchmark config ────────────────────────────────────────────────────────────
ALPHA    = 0.1
L1_RATIO = 0.5
MAX_ITER = 1000
TOL      = 1e-4
N_REPS   = 5

DEFAULT_SIZES = [(10_000, 50), (50_000, 50), (100_000, 50), (100_000, 100)]
LARGE_SIZES   = [(500_000, 50), (1_000_000, 20)]   # 100 MB, 80 MB — spill past L3


# ─── worker (runs inside subprocess after the right .so is installed) ───────────

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

    fit_t = _timeit(
        lambda: rb.elastic_net_fit(X, y, ALPHA, L1_RATIO, MAX_ITER, TOL, True), n_reps)
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


def _run_worker():
    a = sys.argv
    n_rows = int(a[a.index("--n-rows") + 1])
    n_cols = int(a[a.index("--n-cols") + 1])
    n_reps = int(a[a.index("--n-reps") + 1])
    if "--sklearn" in a:
        _worker_sklearn(n_rows, n_cols, n_reps)
    else:
        _worker_rust(n_rows, n_cols, n_reps)


# ─── subprocess launcher ────────────────────────────────────────────────────────

def _spawn(n_rows: int, n_cols: int, n_reps: int, sklearn: bool = False) -> dict:
    env = dict(os.environ)
    env["SKRUB_RUST"] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker",
           "--n-rows", str(n_rows),
           "--n-cols", str(n_cols),
           "--n-reps", str(n_reps)]
    if sklearn:
        cmd.append("--sklearn")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO_ROOT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-800:])
    return json.loads(r.stdout.strip())


# ─── git / build helpers ────────────────────────────────────────────────────────

def _create_worktree(commit_hash: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix=f"stratum-bench-{commit_hash[:7]}-"))
    r = subprocess.run(
        ["git", "worktree", "add", "--detach", str(tmp), commit_hash],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"git worktree add failed:\n{r.stderr.strip()}")
    return tmp


def _remove_worktree(worktree: Path):
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    shutil.rmtree(worktree, ignore_errors=True)


def _build(rust_dir: Path) -> bool:
    """Build the Rust extension in rust_dir and install into the active venv."""
    # prefer the venv-local maturin so the right interpreter is used
    venv_maturin = REPO_ROOT / ".venv" / "bin" / "maturin"
    candidates = []
    if venv_maturin.exists():
        candidates.append([str(venv_maturin), "develop", "--release"])
    candidates += [
        ["maturin", "develop", "--release"],
        ["uv", "run", "maturin", "develop", "--release"],
    ]
    for cmd in candidates:
        r = subprocess.run(cmd, cwd=rust_dir, capture_output=True, text=True)
        if r.returncode == 0:
            return True
    print(f"\n    Build failed. stderr:\n{r.stderr[-600:]}")
    return False


# ─── cache helpers ──────────────────────────────────────────────────────────────

def _key(tag: str, n_rows: int, n_cols: int) -> str:
    return f"{tag}__{n_rows}__{n_cols}"


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


# ─── plotting ───────────────────────────────────────────────────────────────────

def make_plot(cache: dict, sizes: list[tuple[int, int]]):
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
    })

    n = len(sizes)
    xs = list(range(n))
    mb_vals     = [r * c * 4 / 1e6 for r, c in sizes]
    size_labels = [f"{r // 1000}k×{c}\n({mb:.0f} MB)"
                   for (r, c), mb in zip(sizes, mb_vals)]

    commit_colors = [plt.cm.tab10(i) for i in range(len(COMMITS))]
    sklearn_color = "#999999"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=False)
    fig.suptitle(
        "ElasticNet: time per optimization commit vs sklearn",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for phase, ax in zip(("fit", "predict"), axes):

        # sklearn reference
        sk_times = [cache.get(_key("sklearn", r, c), {}).get(phase) for r, c in sizes]
        valid = [(x, t) for x, t in zip(xs, sk_times) if t is not None]
        if valid:
            vx, vt = zip(*valid)
            ax.plot(vx, [t * 1000 for t in vt],
                    color=sklearn_color, linestyle="--", linewidth=2,
                    marker="s", markersize=5, label="sklearn", zorder=2)

        # each commit
        prev_times = None
        for i, commit in enumerate(COMMITS):
            h = commit["hash"]
            times = [cache.get(_key(h, r, c), {}).get(phase) for r, c in sizes]
            valid = [(x, t) for x, t in zip(xs, times) if t is not None]
            if not valid:
                continue
            vx, vt = zip(*valid)
            vt_ms = [t * 1000 for t in vt]
            ax.plot(vx, vt_ms,
                    color=commit_colors[i], linewidth=2.2, marker="o", markersize=6,
                    label=commit["label"].replace("\n", " "), zorder=3)

            # annotate improvement over previous commit (last size only)
            if prev_times is not None:
                for j, (t_cur, t_prev) in enumerate(zip(times, prev_times)):
                    if t_cur and t_prev and t_prev > t_cur:
                        ratio = t_prev / t_cur
                        # only annotate if improvement is meaningful (>5%)
                        if ratio > 1.05:
                            ax.annotate(
                                f"{ratio:.1f}×↑",
                                xy=(xs[j], t_cur * 1000),
                                xytext=(xs[j] + 0.08, t_cur * 1000 * 0.75),
                                fontsize=7.5,
                                color=commit_colors[i],
                                arrowprops=dict(arrowstyle="-", color=commit_colors[i],
                                                lw=0.8),
                            )
            prev_times = times

        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(size_labels)
        ax.set_xlabel("Dataset size", labelpad=8)
        ax.set_ylabel("Median time  (ms)", labelpad=8)
        ax.set_title(f"{'Fit' if phase == 'fit' else 'Predict'} time", pad=8)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.22, which="both")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.2f}"))
        ax.set_xlim(-0.4, n - 0.6)

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_FILE, bbox_inches="tight")
    print(f"Plot → {PLOT_FILE.relative_to(REPO_ROOT)}")


# ─── main ───────────────────────────────────────────────────────────────────────

def main():
    if "--worker" in sys.argv:
        _run_worker()
        return

    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plot-only", action="store_true",
                   help="skip benchmarks, regenerate plot from cache")
    p.add_argument("--refresh",   action="store_true",
                   help="ignore cache, re-run everything")
    p.add_argument("--reps",      type=int, default=N_REPS,
                   help=f"timed repetitions per cell (default {N_REPS})")
    p.add_argument("--large",     action="store_true",
                   help="add large sizes (500k, 1M rows) to expose DRAM-bandwidth effects")
    args = p.parse_args()

    sizes = DEFAULT_SIZES + (LARGE_SIZES if args.large else [])
    cache = {} if args.refresh else load_cache()

    if not args.plot_only:
        # ── sklearn baseline (no rebuild needed) ───────────────────────────────
        sk_missing = [(r, c) for r, c in sizes if _key("sklearn", r, c) not in cache]
        if sk_missing:
            print("── sklearn baseline ──")
            for n_rows, n_cols in sk_missing:
                mb = n_rows * n_cols * 4 / 1e6
                print(f"  {n_rows:>8,}×{n_cols}  ({mb:.0f} MB) ...", end=" ", flush=True)
                try:
                    res = _spawn(n_rows, n_cols, args.reps, sklearn=True)
                    cache[_key("sklearn", n_rows, n_cols)] = res
                    save_cache(cache)
                    print(f"fit={res['fit']*1000:.1f}ms  predict={res['predict']*1000:.1f}ms")
                except Exception as e:
                    print(f"FAILED: {e}")

        # ── per-commit benchmarks ──────────────────────────────────────────────
        for commit in COMMITS:
            h     = commit["hash"]
            label = commit["label"].replace("\n", " ")
            missing = [(r, c) for r, c in sizes if _key(h, r, c) not in cache]

            if not missing:
                print(f"\n── {h} ({label}): all {len(sizes)} sizes cached ──")
                continue

            print(f"\n── {h} ({label}): {len(missing)} size(s) to measure ──")
            print(f"   Creating worktree and building ...", end=" ", flush=True)
            worktree = None
            try:
                worktree = _create_worktree(h)
                ok = _build(worktree / "_rust")
                if not ok:
                    print("build FAILED — skipping this commit")
                    continue
                print("done")

                for n_rows, n_cols in missing:
                    mb = n_rows * n_cols * 4 / 1e6
                    print(f"  {n_rows:>8,}×{n_cols}  ({mb:.0f} MB) ...", end=" ", flush=True)
                    try:
                        res = _spawn(n_rows, n_cols, args.reps)
                        cache[_key(h, n_rows, n_cols)] = res
                        save_cache(cache)
                        sk_fit = cache.get(_key("sklearn", n_rows, n_cols), {}).get("fit")
                        sp = f"  ({sk_fit/res['fit']:.1f}× sklearn)" if sk_fit else ""
                        print(f"fit={res['fit']*1000:.1f}ms  "
                              f"predict={res['predict']*1000:.1f}ms{sp}")
                    except Exception as e:
                        print(f"FAILED: {e}")

            finally:
                if worktree:
                    _remove_worktree(worktree)

        # ── rebuild HEAD so the repo is left in working state ─────────────────
        print(f"\n── Rebuilding HEAD extension ──")
        head_ok = _build(REPO_ROOT / "_rust")
        print("done" if head_ok else "FAILED — run: cd _rust && maturin develop --release")

    if cache:
        make_plot(cache, sizes)
    else:
        print("No cached results — nothing to plot.")


if __name__ == "__main__":
    main()
