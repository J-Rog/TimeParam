#!/usr/bin/env python3
# parse_infractions.py
# Usage:
#   python parse_infractions.py table.txt --out_csv infractions.csv --out_png infractions.png

import re
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
      NRA 0ms 0.75    43
      RA  50ms 1      12
    Returns DataFrame with columns: cfg, lat_ms, res, infractions
    """
    rows = []
    pat = re.compile(r'^(NRA|RA)\s+(\d+)\s*ms\s+([0-9.]+)\s+(\d+)\s*$', re.IGNORECASE)
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("table for"):
            continue
        m = pat.match(line)
        if not m:
            # try generic tokenization fallback
            toks = re.split(r'\s+', line)
            if len(toks) >= 4 and toks[1].lower().endswith("ms"):
                cfg = toks[0].upper()
                lat_ms = int(toks[1].lower().replace("ms",""))
                res = float(toks[2])
                val = int(toks[3])
            else:
                # skip unparseable lines rather than crash
                continue
        else:
            cfg = m.group(1).upper()
            lat_ms = int(m.group(2))
            res = float(m.group(3))
            val = int(m.group(4))
        rows.append({"cfg": cfg, "lat_ms": lat_ms, "res": res, "infractions": val})
    if not rows:
        raise ValueError("No valid rows parsed.")
    df = pd.DataFrame(rows).sort_values(["lat_ms","cfg","res"]).reset_index(drop=True)
    return df


def make_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to rows=(lat_ms,cfg), cols=res, values=infractions
    """
    piv = df.pivot(index=["lat_ms","cfg"], columns="res", values="infractions").sort_index()
    piv = piv.rename_axis(index=["lat_ms","cfg"], columns="resolution").reset_index()
    return piv


def plot_grouped(df: pd.DataFrame, out_png: Path | None):
    """
    Grouped bars: blocks = latencies (0 ms, 50 ms), within each block cfg = [NRA, RA],
    bars per group = resolutions (0.75, 1.0).
    """
    # Ensure consistent order
    lat_order = sorted(df["lat_ms"].unique())
    cfg_order = ["NRA","RA"]
    res_order = sorted(df["res"].unique())  # expects [0.75, 1.0]

    # Build arrays in order
    def get_vals(lat):
        vals_075, vals_100 = [], []
        for cfg in cfg_order:
            vmap = {r: v for r, v in zip(
                df[(df.lat_ms==lat)&(df.cfg==cfg)]["res"].values,
                df[(df.lat_ms==lat)&(df.cfg==cfg)]["infractions"].values
            )}
            vals_075.append(vmap.get(res_order[0], np.nan))
            vals_100.append(vmap.get(res_order[1], np.nan))
        return np.array(vals_075, dtype=float), np.array(vals_100, dtype=float)

    x_base = np.arange(len(cfg_order), dtype=float)
    gap = 2.0
    w = 0.35

    plt.figure(figsize=(9,5.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]
    c075, c100 = colors[0], colors[1]

    x_positions = []
    xticklabels = []
    for i, lat in enumerate(lat_order):
        x = x_base + i*gap
        v075, v100 = get_vals(lat)
        plt.bar(x - w/2, v075, width=w, label="0.75×" if i==0 else None, color=c075)
        plt.bar(x + w/2, v100, width=w, label="1.0×"  if i==0 else None, color=c100)
        for j in range(len(cfg_order)):
            if not np.isnan(v075[j]): plt.text(x[j]-w/2, v075[j], f"{int(v075[j])}", ha="center", va="bottom", fontsize=12)
            if not np.isnan(v100[j]): plt.text(x[j]+w/2, v100[j], f"{int(v100[j])}", ha="center", va="bottom", fontsize=12)
        x_positions.extend(list(x))
        xticklabels.extend([f"{lat} ms\n{cfg}" for cfg in cfg_order])

    plt.xticks(x_positions, xticklabels)
    plt.ylabel("Total traffic-light infractions")
    plt.title("Total traffic-light infractions")
    plt.grid(True, axis="y", alpha=0.2)
    plt.legend(title="Resolution", loc="upper right")
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Parse traffic-light infraction table and plot.")
    ap.add_argument("infile", type=Path, help="Path to text file with rows like: 'NRA 0ms 0.75    43'")
    ap.add_argument("--out_csv", type=Path, default=None, help="Optional path to write parsed CSV")
    ap.add_argument("--out_png", type=Path, default=None, help="Optional path to write bar chart PNG")
    args = ap.parse_args()

    text = args.infile.read_text(encoding="utf-8")
    df = parse_table(text)

    print("\nParsed rows:")
    print(df)

    piv = make_pivot(df)
    print("\nPivot (rows = latency, cfg; cols = resolution):")
    print(piv.to_string(index=False))

    if args.out_csv:
        piv.to_csv(args.out_csv, index=False)
        print(f"\nWrote CSV: {args.out_csv}")

    plot_grouped(df, args.out_png)
    if args.out_png:
        print(f"Wrote PNG: {args.out_png}")


if __name__ == "__main__":
    main()