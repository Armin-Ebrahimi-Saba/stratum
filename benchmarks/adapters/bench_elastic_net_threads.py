#!/usr/bin/env python3
"""
Sweep ElasticNet fit/predict across thread counts and dataset sizes.

Each (thread_count, size) entry is cached in
  benchmarks/results/elastic_net_threads.json
so re-runs skip already-measured combinations.

A comparison plot (speedup over sklearn) is written to
  benchmarks/results/elastic_net_threads_plot.pdf

Usage (from repo root):
    uv run python benchmarks/adapters/bench_elastic_net_threads.py
    uv run python benchmarks/adapters/bench_elastic_net_threads.py --large
    uv run python benchmarks/adapters/bench_elastic_net_threads.py --plot-only
    uv run python benchmarks/adapters/bench_elastic_net_threads.py --refresh
    uv run python benchmarks/adapters/bench_elastic_net_threads.py --no-verify
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── correctness-check config ───────────────────────────────────────────────────
# Tighter convergence than the benchmark so both solvers reach the same minimum.
VERIFY_MAX_ITER = 10_000
VERIFY_TOL      = 1e-6
VERIFY_RTOL     = 0.05   # max allowed relative error on predictions

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
RESULTS    = REPO_ROOT / "benchmarks" / "results"
CACHE_FILE = RESULTS / "elastic_net_threads.json"
PLOT_FILE  = RESULTS / "elastic_net_threads_plot.pdf"

# ── benchmark config ───────────────────────────────────────────────────────────
ALPHA    = 0.1
L1_RATIO = 0.5
MAX_ITER = 1000
TOL      = 1e-4
N_REPS   = 5

DEFAULT_SIZES = [(10_000, 50), (50_000, 50), (100_000, 50), (100_000, 100)]
LARGE_SIZES   = [(500_000, 50), (1_000_000, 20)]

_n_cpu        = os.cpu_count() or 4
THREAD_COUNTS = sorted(set(t for t in [1, 2, 4, 8, _n_cpu] if t <= _n_cpu))


# ─── worker helpers (run inside a subprocess) ──────────────────────────────────

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

    X, y = _make_data(n_rows, n_cols, np.float64)  # sklearn requires f64
    kw = dict(alpha=ALPHA, l1_ratio=L1_RATIO, max_iter=MAX_ITER, tol=TOL, fit_intercept=True)

    for _ in range(2):
        ElasticNet(**kw).fit(X, y)
    gc.collect()

    fit_t = _timeit(lambda: ElasticNet(**kw).fit(X, y), n_reps)
    sk    = ElasticNet(**kw).fit(X, y)

    for _ in range(2):
        sk.predict(X)
    gc.collect()

    pred_t = _timeit(lambda: sk.predict(X), n_reps)
    print(json.dumps({"fit": fit_t, "predict": pred_t}))


def _worker_verify(n_rows: int, n_cols: int):
    """
    Fit both sklearn and Rust to convergence, compare predictions and
    coefficients, and print a JSON result dict.
    Uses tighter convergence (VERIFY_MAX_ITER / VERIFY_TOL) so both solvers
    reach the global minimum before we compare.
    """
    import numpy as np
    from sklearn.linear_model import ElasticNet
    from stratum import _rust_backend as rb

    X_f32, y_f32 = _make_data(n_rows, n_cols, np.float32)
    X_f64 = X_f32.astype(np.float64)
    y_f64 = y_f32.astype(np.float64)

    # ── sklearn reference (f64, well-converged) ────────────────────────────────
    sk_kw = dict(alpha=ALPHA, l1_ratio=L1_RATIO,
                 max_iter=VERIFY_MAX_ITER, tol=VERIFY_TOL, fit_intercept=True)
    sk = ElasticNet(**sk_kw).fit(X_f64, y_f64)
    sk_preds = sk.predict(X_f64).astype(np.float32)
    sk_coef  = sk.coef_.astype(np.float32)

    # ── Rust (f32, same convergence budget) ───────────────────────────────────
    mid, coef_ru, intercept_ru, n_iter = rb.elastic_net_fit(
        X_f32, y_f32, ALPHA, L1_RATIO, VERIFY_MAX_ITER, VERIFY_TOL, True)
    ru_preds = rb.elastic_net_predict(mid, X_f32)

    # ── compare predictions ────────────────────────────────────────────────────
    # Use absolute tolerance in the denominator to avoid division-by-zero on
    # near-zero predictions; mirrors np.testing.assert_allclose semantics.
    abs_err  = np.abs(ru_preds - sk_preds)
    rel_err  = abs_err / (np.abs(sk_preds) + 1e-6)
    max_rel  = float(rel_err.max())
    mean_rel = float(rel_err.mean())

    # ── compare coefficients ───────────────────────────────────────────────────
    coef_mse      = float(np.mean((coef_ru - sk_coef) ** 2))
    max_coef_diff = float(np.abs(coef_ru - sk_coef).max())

    passed = max_rel <= VERIFY_RTOL

    print(json.dumps({
        "passed":          passed,
        "max_pred_rel_err":  max_rel,
        "mean_pred_rel_err": mean_rel,
        "coef_mse":          coef_mse,
        "max_coef_diff":     max_coef_diff,
        "rust_n_iter":       n_iter,
        "sklearn_n_iter":    sk.n_iter_,
    }))


def _run_worker():
    """Entry point when called with --worker. Reads params from sys.argv."""
    a = sys.argv
    n_rows = int(a[a.index("--n-rows") + 1])
    n_cols = int(a[a.index("--n-cols") + 1])

    if "--verify" in a:
        _worker_verify(n_rows, n_cols)
        return

    n_reps = int(a[a.index("--n-reps") + 1])
    if "--sklearn" in a:
        _worker_sklearn(n_rows, n_cols, n_reps)
    else:
        _worker_rust(n_rows, n_cols, n_reps)


# ─── subprocess launcher ───────────────────────────────────────────────────────

def _spawn(n_rows: int, n_cols: int, n_reps: int,
           n_threads: int | None = None) -> dict:
    """Spawn a worker subprocess; return the parsed JSON result."""
    env = dict(os.environ)
    env["SKRUB_RUST"] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker",
           "--n-rows", str(n_rows),
           "--n-cols", str(n_cols),
           "--n-reps", str(n_reps)]

    if n_threads is None:
        cmd.append("--sklearn")
    else:
        env["SKRUB_RUST_THREADS"] = str(n_threads)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout.strip())


# ─── correctness-check runner ─────────────────────────────────────────────────

def _run_verify(n_rows: int, n_cols: int) -> dict:
    """Spawn a worker subprocess for the correctness check; return parsed result."""
    env = dict(os.environ)
    env["SKRUB_RUST"] = "1"
    env["SKRUB_RUST_THREADS"] = "1"   # single thread; correctness is thread-independent
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker", "--verify",
           "--n-rows", str(n_rows),
           "--n-cols", str(n_cols)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout.strip())


# ─── cache helpers ─────────────────────────────────────────────────────────────

def _key(tag: str, n_rows: int, n_cols: int) -> str:
    return f"{tag}__{n_rows}__{n_cols}"


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


# ─── plotting ──────────────────────────────────────────────────────────────────

def make_plot(cache: dict, sizes: list[tuple[int, int]], thread_counts: list[int]):
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
    })

    n = len(sizes)
    xs          = list(range(n))
    mb_vals     = [r * c * 4 / 1e6 for r, c in sizes]
    size_labels = [f"{r // 1000}k×{c}\n({mb:.0f} MB)"
                   for (r, c), mb in zip(sizes, mb_vals)]

    # pick distinct colours; first colour reserved for sklearn baseline marker
    colors = [plt.cm.tab10(i) for i in range(len(thread_counts))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    fig.suptitle("ElasticNet: Rust vs sklearn speedup across dataset sizes",
                 fontsize=14, fontweight="bold", y=1.01)

    for phase, ax in zip(("fit", "predict"), axes):
        sk_times = [cache.get(_key("sklearn", r, c), {}).get(phase) for r, c in sizes]

        for i, n_t in enumerate(thread_counts):
            tag   = f"rust_{n_t}"
            label = f"Rust {n_t}T" if n_t < _n_cpu else f"Rust {n_t}T (all cores)"
            ru_times = [cache.get(_key(tag, r, c), {}).get(phase) for r, c in sizes]

            valid_xs, speedups = [], []
            for j, (ru, sk) in enumerate(zip(ru_times, sk_times)):
                if ru is not None and sk is not None and ru > 0:
                    valid_xs.append(xs[j])
                    speedups.append(sk / ru)

            if speedups:
                ax.plot(valid_xs, speedups, marker="o", color=colors[i],
                        label=label, linewidth=2, markersize=6, zorder=3)

        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.2,
                   label="sklearn baseline (1×)", zorder=2)

        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(size_labels)
        ax.set_xlabel("Dataset size", labelpad=8)
        ax.set_ylabel("Speedup over sklearn  (×)", labelpad=8)
        ax.set_title(f"{'Fit' if phase == 'fit' else 'Predict'} time", pad=8)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.25, which="both")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.1f}×" if v < 10 else f"{v:.0f}×"))
        ax.set_xlim(-0.4, n - 0.6)

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_FILE, bbox_inches="tight")
    print(f"Plot → {PLOT_FILE.relative_to(REPO_ROOT)}")


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    # Worker mode: must be detected before argparse to avoid importing heavy deps
    if "--worker" in sys.argv:
        _run_worker()
        return

    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--large",     action="store_true", help="add large sizes (500k, 1M rows)")
    p.add_argument("--plot-only", action="store_true", help="skip benchmarks, regenerate plot only")
    p.add_argument("--refresh",   action="store_true", help="ignore cache, re-run everything")
    p.add_argument("--reps",      type=int, default=N_REPS, help=f"timed reps (default {N_REPS})")
    p.add_argument("--no-verify", action="store_true",
                   help="skip correctness checks against sklearn")
    args = p.parse_args()

    sizes = DEFAULT_SIZES + (LARGE_SIZES if args.large else [])
    cache = {} if args.refresh else load_cache()

    if not args.plot_only:
        # ── correctness checks (one per size, before timing) ──────────────────
        if not args.no_verify:
            print("── Correctness checks "
                  f"(rtol={VERIFY_RTOL}, max_iter={VERIFY_MAX_ITER}, tol={VERIFY_TOL}) ──")
            any_failed = False
            for n_rows, n_cols in sizes:
                mb  = n_rows * n_cols * 4 / 1e6
                vk  = _key("verify", n_rows, n_cols)
                tag = f"{n_rows:>8,}×{n_cols:<4}  ({mb:.0f} MB)"

                if vk in cache:
                    v = cache[vk]
                    status = "PASS ✓" if v["passed"] else "FAIL ✗"
                    print(f"  {tag}  {status}  (cached)"
                          f"  max_rel={v['max_pred_rel_err']:.2e}"
                          f"  coef_mse={v['coef_mse']:.2e}")
                    if not v["passed"]:
                        any_failed = True
                    continue

                print(f"  {tag}  checking ...", end=" ", flush=True)
                try:
                    v = _run_verify(n_rows, n_cols)
                    cache[vk] = v
                    save_cache(cache)
                    status = "PASS ✓" if v["passed"] else "FAIL ✗"
                    print(f"{status}"
                          f"  max_rel={v['max_pred_rel_err']:.2e}"
                          f"  coef_mse={v['coef_mse']:.2e}"
                          f"  iters rust={v['rust_n_iter']} sk={v['sklearn_n_iter']}")
                    if not v["passed"]:
                        any_failed = True
                except Exception as e:
                    print(f"ERROR: {e}")
                    any_failed = True

            if any_failed:
                print("\nCorrectness check(s) failed — aborting benchmark.")
                print("Fix the implementation or re-run with --no-verify to skip checks.")
                sys.exit(1)
            print()

        # ── timing benchmark ──────────────────────────────────────────────────
        jobs: list[tuple[str, str, int | None]] = [("sklearn", "sklearn", None)]
        for n_t in THREAD_COUNTS:
            jobs.append((f"rust-{n_t}T", f"rust_{n_t}", n_t))

        total = len(sizes) * len(jobs)
        done  = 0

        print(f"── Timing benchmark ({total} cells) ──")
        for n_rows, n_cols in sizes:
            mb = n_rows * n_cols * 4 / 1e6
            header = f"{n_rows:>8,}×{n_cols:<4}  ({mb:.0f} MB)"

            for label, tag, n_threads in jobs:
                done += 1
                k = _key(tag, n_rows, n_cols)
                if k in cache:
                    print(f"  [{done:>{len(str(total))}}/{total}] {label:<12} {header}  cached")
                    continue

                print(f"  [{done:>{len(str(total))}}/{total}] {label:<12} {header} ...",
                      end=" ", flush=True)
                try:
                    res = _spawn(n_rows, n_cols, args.reps, n_threads)
                    cache[k] = res
                    save_cache(cache)

                    sk_fit = cache.get(_key("sklearn", n_rows, n_cols), {}).get("fit")
                    sp_str = (f"  {sk_fit / res['fit']:.1f}× fit" if sk_fit and n_threads else "")
                    print(f"fit={res['fit']*1000:7.1f}ms  predict={res['predict']*1000:7.1f}ms{sp_str}")
                except Exception as e:
                    print(f"FAILED: {e}")

    if cache:
        make_plot(cache, sizes, THREAD_COUNTS)
    else:
        print("No cached results to plot.")


if __name__ == "__main__":
    main()
