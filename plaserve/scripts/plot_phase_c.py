"""Plot Phase C throughput, latency and LAPS/vanilla speedup from a campaign directory.

Usage: python scripts/plot_phase_c.py results/phase-c-20261027 --clean 1 2 5 6 --flagged 3 4
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES = ("laps", "vanilla_sglang", "disaggregation")
NAMES = {"laps": "LAPS", "vanilla_sglang": "Vanilla SGLang", "disaggregation": "Disaggregation"}
COLORS = {"laps": "#2a78d6", "vanilla_sglang": "#eb6834", "disaggregation": "#1baf7a"}
CCS = (1, 4, 8, 16)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(campaign, repeat):
    rows = json.loads((campaign / f"repeat-{repeat}" / "validation.json").read_text())["results"]
    return {(r["policy"], r["concurrency"]): r for r in rows}


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_xlabel("Concurrent requests", color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)


def grouped_bars(path, runs, metric, title, ylabel, fmt):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    width = 0.26
    for i, policy in enumerate(POLICIES):
        xs = [c + (i - 1) * width for c in range(len(CCS))]
        values = [[run[policy, cc][metric] for run in runs] for cc in CCS]
        means = [statistics.mean(v) for v in values]
        stds = [statistics.stdev(v) for v in values]
        ax.bar(xs, means, width - 0.03, color=COLORS[policy], label=NAMES[policy],
               yerr=stds, error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2))
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + s, fmt.format(m), ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(CCS)), [str(c) for c in CCS])
    style(ax, title, ylabel)
    ax.legend(frameon=False, ncols=3, loc="upper left", labelcolor=INK)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def speedup(path, clean, flagged):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
    ax.text(len(CCS) - 0.5, 1.005, "vanilla = 1.0", ha="right", va="bottom", fontsize=8, color=INK2)
    for runs, filled, label in ((clean, True, "Clean repeat"), (flagged, False, "Repeat sharing GPUs 2-3")):
        for k, run in enumerate(runs):
            ratios = [run["laps", cc]["request_throughput_req_s"] / run["vanilla_sglang", cc]["request_throughput_req_s"]
                      for cc in CCS]
            offsets = [c + (k - (len(runs) - 1) / 2) * 0.07 for c in range(len(CCS))]
            ax.scatter(offsets, ratios, s=40, zorder=3, linewidths=1.5, edgecolors=COLORS["laps"],
                       facecolors=COLORS["laps"] if filled else SURFACE, label=label if k == 0 else None)
    means = [statistics.mean(r["laps", cc]["request_throughput_req_s"] / r["vanilla_sglang", cc]["request_throughput_req_s"]
                             for r in clean) for cc in CCS]
    for c, m in enumerate(means):
        ax.hlines(m, c - 0.25, c + 0.25, color=INK, linewidth=2, zorder=4)
        ax.text(c + 0.28, m, f"{m:.2f}×", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(CCS)), [str(c) for c in CCS])
    ax.set_xlim(-0.5, len(CCS) - 0.3)
    style(ax, "LAPS throughput relative to vanilla SGLang (bar = mean of clean repeats)", "LAPS / vanilla req/s")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("campaign", type=Path)
    p.add_argument("--clean", type=int, nargs="+", required=True)
    p.add_argument("--flagged", type=int, nargs="*", default=[])
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    out = a.output or a.campaign / "figures"
    out.mkdir(parents=True, exist_ok=True)
    clean = [load(a.campaign, n) for n in a.clean]
    flagged = [load(a.campaign, n) for n in a.flagged]
    grouped_bars(out / "throughput.png", clean, "request_throughput_req_s",
                 "Request throughput, Qwen2.5-7B, 1 prefill + 1 decode (TP2 each)", "Requests / s", "{:.0f}")
    grouped_bars(out / "latency.png", clean, "mean_ttft_ms",
                 "Mean request latency (one output token)", "Milliseconds", "{:.1f}")
    speedup(out / "speedup.png", clean, flagged)
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
