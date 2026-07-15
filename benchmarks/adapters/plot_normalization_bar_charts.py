#!/usr/bin/env python3
"""
Bar charts comparing sklearn vs Rust for every normalization kernel.

One PNG per input size (one plot per file):
  normalization_bars_<rows>x<cols>.png      e.g. normalization_bars_10000x100.png

Each chart is a single input size; the x-axis lists the kernels, red = sklearn,
blue = Rust, and the label above each Rust bar is the speedup over sklearn.

Reads benchmarks/results/normalization_full_cache.json (produced by
bench_normalization_charts.py). Rust numbers are taken at a fixed thread count
(default: the largest thread count present in the cache).

Usage:
    uv run python benchmarks/adapters/plot_normalization_bar_charts.py
    uv run python benchmarks/adapters/plot_normalization_bar_charts.py --threads 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_normalization_charts import (          # noqa: E402
    KERNELS, KERNEL_LABEL, _rkey, _skey, _n_cpu,
)

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
RESULTS    = REPO_ROOT / "benchmarks" / "results"
CACHE_FILE = RESULTS / "normalization_full_cache.json"

SKLEARN_COLOR = "#CC2222"
RUST_COLOR    = "#2255CC"
BAR_W         = 0.38

ALL_KERNEL_KEYS = [k for k, *_ in KERNELS]


def size_title(r: int, c: int) -> str:
    mb = r * c * 4 / 1e6
    k = r // 1000
    rows = f"{k // 1000}M" if k >= 1000 else f"{k}k"
    return f"{rows}×{c}  ({mb:.0f} MB)"


def all_sizes(cache: dict) -> list[tuple[int, int]]:
    sizes = {tuple(map(int, k.rsplit("__", 2)[-2:]))
             for k in cache if "__t" in k and not k.startswith("sklearn__")}
    return sorted(sizes)


def pick_threads(cache: dict, requested: int | None) -> int:
    present = sorted({int(k.split("__t")[1].split("__")[0])
                      for k in cache if "__t" in k and not k.startswith("sklearn__")})
    if not present:
        return requested or _n_cpu
    if requested is not None:
        return requested if requested in present else max(present)
    return max(present)


def make_size_figure(cache: dict, r: int, c: int, n_t: int) -> bool:
    """One PNG for a single input size; returns True if it had data."""
    kernels, sk_ms, ru_ms = [], [], []
    for k in ALL_KERNEL_KEYS:
        sk = cache.get(_skey(k, r, c), {}).get("time")
        ru = cache.get(_rkey(k, n_t, r, c), {}).get("time")
        if sk is None or ru is None:
            continue
        kernels.append(k); sk_ms.append(sk * 1000); ru_ms.append(ru * 1000)
    if not kernels:
        return False

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(kernels)), 5.5))
    x = np.arange(len(kernels))
    ax.bar(x - BAR_W/2, sk_ms, BAR_W, color=SKLEARN_COLOR, label="sklearn", zorder=3)
    ax.bar(x + BAR_W/2, ru_ms, BAR_W, color=RUST_COLOR,    label="rust",    zorder=3)

    for xi, (sk, ru) in enumerate(zip(sk_ms, ru_ms)):
        if ru > 0:
            ax.text(x[xi] + BAR_W/2, ru * 1.04, f"{sk/ru:.1f}×",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color="#1a3d99")

    ax.set_xticks(x)
    ax.set_xticklabels([KERNEL_LABEL[k] for k in kernels], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Median time (ms)")
    ax.set_title(f"Normalization: sklearn vs Rust — {size_title(r, c)}  "
                 f"[{n_t} threads, labels = speedup]", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.2f}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    out = RESULTS / f"normalization_bars_{r}x{c}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → {out.relative_to(REPO_ROOT)}")
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=None,
                    help="thread count whose Rust numbers to plot (default: max in cache)")
    args = ap.parse_args()

    if not CACHE_FILE.exists():
        sys.exit(f"No cache at {CACHE_FILE}. Run bench_normalization_charts.py first.")
    cache = json.loads(CACHE_FILE.read_text())

    n_t = pick_threads(cache, args.threads)
    print(f"Plotting Rust numbers at {n_t} threads — one file per size.")
    for r, c in all_sizes(cache):
        make_size_figure(cache, r, c, n_t)
    print("Done.")


if __name__ == "__main__":
    main()
