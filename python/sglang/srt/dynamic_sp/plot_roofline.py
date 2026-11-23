# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""Individual roofline figures from the SP attention probe.

Plot labels use processed tokens for the existing KV prefix and remaining tokens
for all new tokens processed in the measured step. Raw history/chunk fields are
retained for compatibility; the probe processes all remaining tokens in one step.

F is useful causal QK+AV FLOPs per GPU, M is compulsory Q/K/V/output bytes,
and V is bytes sent per GPU. P and B are median empirical GEMM and copy rates.
The memory roof is min(P, B*F/M). Each network roof uses the maximum observed
V/time for its SP degree, including concurrent receive, with no startup model.
The blocking roof is F/(max(F/P, M/B) + measured isolated transport time).
For production overlap, the optimistic reference is F/max(F/P, M/B, Ttransfer).
These models exclude extra DRAM traffic, softmax, launch overhead, and accumulator
costs. Isolated transfer time is not a measurement of exposed communication stall.
Each sample uses the median of per-repetition maximum rank durations; raw
repetitions are retained. Shading shows empirical p10-p90, not confidence intervals. Hardware rates are empirical references, not peaks.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.samples.read_text())
    out = args.samples.parent
    rows = raw["samples"]
    production = raw["metadata"].get("execution") == "production-overlap"
    degrees = sorted({r["degree"] for r in rows})
    histories = sorted({r["history"] for r in rows})
    colors = {d: f"C{i}" for i, d in enumerate(degrees)}
    compute = float(np.median([h["compute_flops_s"] for h in raw["hardware"]]))
    memory = float(np.median([h["memory_bytes_s"] for h in raw["hardware"]]))
    # Use observed sustained rates. A latency/bandwidth regression on four
    # message sizes is sensitive to startup outliers and can invent a false roof.
    bandwidths = {
        d: max(
            p["bytes_per_rank"] / p["seconds"]
            for p in raw["transport"]
            if p["degree"] == d
        )
        for d in degrees
        if d > 1
    }
    for r in rows:
        times = r.get("critical_repetitions_seconds", [r["seconds"]])
        r["seconds_p10"], r["seconds_p90"] = map(float, np.percentile(times, [10, 90]))
        f = r["local_flops_mean"]
        r["memory_intensity"] = f / r["compulsory_memory_bytes"]
        r["network_intensity"] = (
            f / r["network_send_bytes"] if r["network_send_bytes"] else None
        )
        r["achieved_tflops"] = f / r["seconds"] / 1e12
        r["ideal_local_seconds"] = max(
            f / compute, r["compulsory_memory_bytes"] / memory
        )
        r["blocking_roof_tflops"] = (
            f / (r["ideal_local_seconds"] + r["transport_seconds"]) / 1e12
        )
        r["overlap_roof_tflops"] = (
            f / max(r["ideal_local_seconds"], r["transport_seconds"]) / 1e12
        )
    with (out / "roofline.csv").open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                k
                for k in rows[0]
                if k not in ("rank_measurements", "critical_repetitions_seconds")
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    (out / "ceilings.json").write_text(
        json.dumps(
            {
                "compute_flops_s": compute,
                "memory_bytes_s": memory,
                "network_bytes_s": bandwidths,
                "network_estimator": "Maximum observed bytes/second over calibrated message sizes",
            },
            indent=2,
        )
        + "\n"
    )

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 2,
            "lines.markersize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200)
        plt.close(fig)

    def log_axes(ax):
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
            axis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:g}")
            )
            axis.set_minor_formatter(NullFormatter())
        ax.grid(which="major", alpha=0.22)
        ax.set_axisbelow(True)

    # One SP degree and history per roofline: no shifting ownership or history
    # hidden in a cloud of points. All ceilings use the same horizontal axis.
    for h in histories:
        for d in degrees:
            points = sorted(
                [r for r in rows if r["degree"] == d and r["history"] == h],
                key=lambda r: r["chunk"],
            )
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            log_axes(ax)
            q = np.geomspace(points[0]["chunk"] / 4, points[-1]["chunk"] * 4, 400)
            heads = raw["metadata"]["query_heads"]
            kv_heads = raw["metadata"]["kv_heads"]
            dim = raw["metadata"]["head_dim"]
            f = 4 * heads * dim / d * (q * h + q * (q + 1) / 2)
            m = 2 * (2 * (h + q) * kv_heads * dim + 2 * (q / d) * heads * dim)
            if d > 1:
                v = (d - 1) * np.ceil((h + q) / d) * 2 * kv_heads * dim * 2
                x = f / v
                key = "network_intensity"
                xlabel = "Communication intensity (FLOP / byte sent)"
            else:
                x = f / m
                key = "memory_intensity"
                xlabel = "Memory intensity (FLOP / compulsory byte)"
            ax.plot(
                x,
                np.minimum(compute, memory * f / m) / 1e12,
                color="0.4",
                label="Compute + memory ceiling",
            )
            if d > 1:
                ax.plot(
                    x,
                    bandwidths[d] * x / 1e12,
                    color="C1",
                    label="Communication bandwidth ceiling",
                )
                ax.plot(
                    [r[key] for r in points],
                    [
                        r[
                            "overlap_roof_tflops"
                            if production
                            else "blocking_roof_tflops"
                        ]
                        for r in points
                    ],
                    "k--",
                    label="Ideal overlap reference"
                    if production
                    else "Ceiling with blocking transfer time",
                )
            ax.plot(
                [r[key] for r in points],
                [r["achieved_tflops"] for r in points],
                "o-",
                color="C0",
                label="Measured overlapped attention"
                if production
                else "Measured attention",
            )
            ax.fill_between(
                [r[key] for r in points],
                [r["local_flops_mean"] / r["seconds_p90"] / 1e12 for r in points],
                [r["local_flops_mean"] / r["seconds_p10"] / 1e12 for r in points],
                color="C0",
                alpha=0.18,
            )
            for r in points:
                if len(points) > 5 and r["chunk"] not in (256, 2048, 16384):
                    continue
                ax.annotate(
                    f"{r['chunk']:,} remaining",
                    (r[key], r["achieved_tflops"]),
                    xytext=(7, -15),
                    textcoords="offset points",
                    fontsize=10,
                )
            ax.set(
                xlabel=xlabel,
                ylabel="Useful attention performance (TFLOP/s per GPU)",
                title=f"{'Overlapped SP attention: ' if production else ''}SP = {d} GPUs\nProcessed tokens = {h:,}",
            )
            ax.set_ylim(
                min(r["achieved_tflops"] for r in points) * 0.45, compute / 1e12 * 2
            )
            ax.set_xlim(points[0][key] * 0.55, points[-1][key] * 3)
            ax.legend(loc="lower right", framealpha=0.95)
            save(fig, f"roofline-sp{d}-history-{h}")

        # Latency is the scheduling quantity. Per-GPU FLOP/s alone cannot show
        # whether increasing degree helps complete a fixed chunk sooner.
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for d in degrees:
            points = sorted(
                [r for r in rows if r["degree"] == d and r["history"] == h],
                key=lambda r: r["chunk"],
            )
            ax.plot(
                [r["chunk"] for r in points],
                [r["seconds"] * 1000 for r in points],
                "o-",
                color=colors[d],
                label=f"SP = {d}",
            )
            ax.fill_between(
                [r["chunk"] for r in points],
                [r["seconds_p10"] * 1000 for r in points],
                [r["seconds_p90"] * 1000 for r in points],
                color=colors[d],
                alpha=0.15,
            )
        ax.set_xscale("log", base=2)
        chunks = sorted({r["chunk"] for r in rows})
        ax.set_xticks(chunks[::2], [f"{q:,}" for q in chunks[::2]], rotation=25)
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.22)
        ax.set_axisbelow(True)
        ax.set(
            xlabel="Remaining tokens (across all SP GPUs)",
            ylabel="Attention time (ms)",
            title=f"Processed tokens = {h:,}",
        )
        ax.legend(title="Lower time is better", ncol=2)
        save(fig, f"chunk-latency-history-{h}")
    if 1 not in degrees or len(degrees) < 2:
        print(f"[roofline] Plots and CSV: {out}")
        return

    from matplotlib.colors import ListedColormap

    chunks = sorted({r["chunk"] for r in rows})
    best = np.zeros((len(histories), len(chunks)), dtype=int)
    choices = []
    for i, h in enumerate(histories):
        for j, q in enumerate(chunks):
            candidates = sorted(
                [r for r in rows if r["history"] == h and r["chunk"] == q],
                key=lambda r: r["seconds"],
            )
            winner = candidates[0]
            best[i, j] = degrees.index(winner["degree"])
            reference = next(r["seconds"] for r in candidates if r["degree"] == 1)
            choices.append(
                {
                    "history": h,
                    "chunk": q,
                    "degree": winner["degree"],
                    "seconds": winner["seconds"],
                    "speedup": reference / winner["seconds"],
                    "runner_up_degree": candidates[1]["degree"],
                    "runner_up_ratio": candidates[1]["seconds"] / winner["seconds"],
                }
            )
    with (out / "best-sp.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(choices[0]))
        writer.writeheader()
        writer.writerows(choices)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.imshow(
        best,
        cmap=ListedColormap(
            [
                "#dbeafe",
                "#ffedd5",
                "#dcfce7",
                "#fce7f3",
                "#ede9fe",
                "#cffafe",
                "#fef9c3",
                "#e2e8f0",
            ][: len(degrees)]
        ),
        vmin=-0.5,
        vmax=len(degrees) - 0.5,
        aspect="auto",
    )
    for i, h in enumerate(histories):
        for j, q in enumerate(chunks):
            c = choices[i * len(chunks) + j]
            ax.text(
                j,
                i,
                f"SP{c['degree']}{'*' if c['runner_up_ratio'] < 1.05 else ''}\n{c['speedup']:.1f}x",
                ha="center",
                va="center",
                fontsize=10,
            )
    ax.set_xticks(
        range(len(chunks)), [f"{q:,}" for q in chunks], rotation=35, ha="right"
    )
    ax.set_yticks(range(len(histories)), [f"{h:,}" for h in histories])
    ax.set(
        xlabel="Remaining tokens; * within 5% of the next-fastest degree",
        ylabel="Processed tokens",
        title="Fastest measured SP degree and speedup over SP=1",
    )
    save(fig, "best-sp")
    print(f"[roofline] Plots and CSV: {out}")


if __name__ == "__main__":
    main()
