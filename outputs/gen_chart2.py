#!/usr/bin/env python3
# parse_lane_stats.py
# Usage:
#   python parse_lane_stats.py lane_table.txt --out_csv lane_stats.csv --out_png lane_stats.png

import re
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,          # base font size
    "axes.titlesize": 16,     # figure/axes titles
    "axes.labelsize": 14,     # x and y labels
    "xtick.labelsize": 12,    # x tick labels
    "ytick.labelsize": 12,    # y tick labels
    "legend.fontsize": 11,    # legend text
    "legend.title_fontsize": 11,
})

def parse_table(text: str) -> pd.DataFrame:
    """
    Parse lines like:
      NRA 0ms 0.75   5     3.857098533
      RA  50ms 1     4.92  3.777432991
    Returns DataFrame with columns: cfg, lat_ms, res, mean, sd
    """
    rows = []
    # strict pattern
    pat = re.compile(
        r'^(NRA|RA)\s+(\d+)\s*ms\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$',
        re.IGNORECASE
    )
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # skip header lines
        if line.lower().startswith("table for"):
            continue
        if line.lower().startswith(("nra", "ra")) and "mean" in line.lower():
            # header row with column names, skip
            continue

        m = pat.match(line)
        if m:
            cfg = m.group(1).upper()
            lat_ms = int(m.group(2))
            res = float(m.group(3))
            mean = float(m.group(4))
            sd = float(m.group(5))
            rows.append({"cfg": cfg, "lat_ms": lat_ms, "res": res, "mean": mean, "sd": sd})
            continue

        # fallback: tokenized parse if spacing is irregular
        toks = re.split(r'\s+', line)
        if len(toks) >= 5 and toks[1].lower().endswith("ms"):
            try:
                cfg = toks[0].upper()
                lat_ms = int(toks[1].lower().replace("ms", ""))
                res = float(toks[2])
                mean = float(toks[3])
                sd = float(toks[4])
                rows.append({"cfg": cfg, "lat_ms": lat_ms, "res": res, "mean": mean, "sd": sd})
                continue
            except ValueError:
                pass  # drop unparseable line

    if not rows:
        raise ValueError("No valid rows parsed from the input text.")
    df = pd.DataFrame(rows).sort_values(["lat_ms", "cfg", "res"]).reset_index(drop=True)
    return df


def make_pivots(df: pd.DataFrame):
    """
    Convenience pivots for quick inspection:
      rows=(lat_ms,cfg), cols=res, values=mean or sd.
    """
    piv_mean = (
        df.pivot(index=["lat_ms", "cfg"], columns="res", values="mean")
          .rename_axis(index=["lat_ms", "cfg"], columns="resolution")
          .reset_index()
    )
    piv_sd = (
        df.pivot(index=["lat_ms", "cfg"], columns="res", values="sd")
          .rename_axis(index=["lat_ms", "cfg"], columns="resolution")
          .reset_index()
    )
    return piv_mean, piv_sd


def plot_grouped_mean_sd(df: pd.DataFrame, out_png: Path | None):
    """
    Grouped bars with error bars:
      blocks = latencies (0 ms, 50 ms),
      within each block cfg = [NRA, RA],
      bars per group = resolutions (0.75, 1.0).
    """
    # Ensure expected orders
    lat_order = sorted(df["lat_ms"].unique())
    cfg_order = ["NRA", "RA"]
    res_order = sorted(df["res"].unique())  # expects [0.75, 1.0]

    # helper to fetch mean/sd arrays in the desired order
    def get_vals(lat):
        m075, m100, s075, s100 = [], [], [], []
        for cfg in cfg_order:
            sub = df[(df.lat_ms == lat) & (df.cfg == cfg)]
            vmap_mean = {r: v for r, v in zip(sub["res"].values, sub["mean"].values)}
            vmap_sd   = {r: v for r, v in zip(sub["res"].values, sub["sd"].values)}
            m075.append(vmap_mean.get(res_order[0], np.nan))
            m100.append(vmap_mean.get(res_order[1], np.nan))
            s075.append(vmap_sd.get(res_order[0], np.nan))
            s100.append(vmap_sd.get(res_order[1], np.nan))
        return np.array(m075, float), np.array(m100, float), np.array(s075, float), np.array(s100, float)

    x_base = np.arange(len(cfg_order), dtype=float)
    gap = 2.0
    w = 0.35
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]
    c075, c100 = colors[0], colors[1]

    plt.figure(figsize=(10, 6))
    x_positions = []
    xticklabels = []

    for i, lat in enumerate(lat_order):
        x = x_base + i * gap
        m075, m100, s075, s100 = get_vals(lat)
        # plot means with sd error bars
        plt.bar(x - w/2, m075, width=w, yerr=s075, capsize=4,
                label="0.75×" if i == 0 else None, color=c075)
        plt.bar(x + w/2, m100, width=w, yerr=s100, capsize=4,
                label="1.0×"  if i == 0 else None, color=c100)
        # labels
        for j in range(len(cfg_order)):
            if not np.isnan(m075[j]):
                plt.text(x[j]-w/2, m075[j] + (0 if np.isnan(s075[j]) else s075[j]) + 0.2,
                         f"{m075[j]:.2f}", ha="center", va="bottom", fontsize=10)
            if not np.isnan(m100[j]):
                plt.text(x[j]+w/2, m100[j] + (0 if np.isnan(s100[j]) else s100[j]) + 0.2,
                         f"{m100[j]:.2f}", ha="center", va="bottom", fontsize=10)

        x_positions.extend(list(x))
        xticklabels.extend([f"{lat} ms\n{cfg}" for cfg in cfg_order])

    plt.xticks(x_positions, xticklabels)
    plt.ylabel("Lane crossings (mean ± sd)")
    plt.title("Lane crossings")
    plt.grid(True, axis="y", alpha=0.2)
    plt.legend(title="Resolution", loc="upper right")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Parse lane mean±sd table and plot.")
    ap.add_argument("infile", type=Path, help="Path to text file with rows like: 'NRA 0ms 0.75  5  3.8571'")
    ap.add_argument("--out_csv", type=Path, default=None, help="Optional path to write parsed CSV (long format)")
    ap.add_argument("--out_png", type=Path, default=None, help="Optional path to write bar chart PNG")
    args = ap.parse_args()

    text = args.infile.read_text(encoding="utf-8")
    df = parse_table(text)

    print("\nParsed rows:")
    print(df)

    piv_mean, piv_sd = make_pivots(df)
    print("\nPivot MEAN (rows = latency, cfg; cols = resolution):")
    print(piv_mean.to_string(index=False))
    print("\nPivot SD (rows = latency, cfg; cols = resolution):")
    print(piv_sd.to_string(index=False))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote CSV: {args.out_csv}")

    plot_grouped_mean_sd(df, args.out_png)
    if args.out_png:
        print(f"Wrote PNG: {args.out_png}")


if __name__ == "__main__":
    main()