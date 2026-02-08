#!/usr/bin/env python3
# plot_resolution_effects.py
# Usage:
#   python plot_resolution_effects.py table.txt --out_csv effects.csv --out_dir figs/

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
    Expected structure:
      table for resolution effects  success rate  collision rate
      NRA 0.75
      0 ms   100   8.35
      50 ms  100   16.65
      ...
      NRA 1.0
      0 ms   100   8.35
      ...

    Returns long-form DataFrame with columns:
      cfg (e.g., 'NRA'/'RA'), res (float), lat_ms (int),
      success_rate (float), collision_rate (float)
    """
    rows = []
    cfg = None
    res = None

    header_pat = re.compile(r'^(NRA|RA)\s+([0-9.]+)\s*$', re.IGNORECASE)
    data_pat   = re.compile(r'^(\d+)\s*ms\s+([0-9.]+)\s+([0-9.]+)\s*$')

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # skip descriptive header
        if line.lower().startswith("table for"):
            continue

        # header: "<CFG> <RES>"
        mh = header_pat.match(line)
        if mh:
            cfg = mh.group(1).upper()
            res = float(mh.group(2))
            continue

        # data row under current header
        md = data_pat.match(line)
        if md and cfg is not None and res is not None:
            lat_ms = int(md.group(1))
            succ = float(md.group(2))
            coll = float(md.group(3))
            rows.append({
                "cfg": cfg, "res": res, "lat_ms": lat_ms,
                "success_rate": succ, "collision_rate": coll
            })

    if not rows:
        raise ValueError("No valid rows parsed. Check input formatting.")
    df = pd.DataFrame(rows).sort_values(["cfg","res","lat_ms"]).reset_index(drop=True)
    return df


def plot_per_cfg(df: pd.DataFrame, out_dir: Path | None):
    """
    One figure per cfg with two stacked subplots:
      Top: success rate vs latency (lines per resolution)
      Bottom: collision rate vs latency (lines per resolution)
    """
    cfgs = df["cfg"].unique()
    # Consistent color mapping by resolution
    res_sorted = sorted(df["res"].unique())
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {res_sorted[i]: colors[i % len(colors)] for i in range(len(res_sorted))}

    for cfg in cfgs:
        sub = df[df["cfg"] == cfg].copy()
        lats = sorted(sub["lat_ms"].unique())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 4.6), sharex=True)
        for res in res_sorted:
            cur = sub[sub["res"] == res].sort_values("lat_ms")
            if cur.empty:
                continue
            x = cur["lat_ms"].to_numpy()
            ax1.plot(x, cur["success_rate"], marker="o", label=f"{res:.2f}×", color=color_map[res])
            ax2.plot(x, cur["collision_rate"], marker="o", label=f"{res:.2f}×", color=color_map[res])

        # Styling
        for ax, ylab in [(ax1, "Success rate (%)"), (ax2, "Collision rate (%)")]:
            ax.set_ylabel(ylab)
            ax.set_ylim(0, 100)
            ax.grid(True, axis="both", alpha=0.25)
            ax.set_xticks(lats)
            ax.set_xticklabels([f"{l} ms" for l in lats])

        ax2.set_xlabel("Execution latency")
        ax1.set_title(f"Latency Effects — {cfg}")
        ax1.legend(
            title="Resolution",
            loc="lower left",   # or "upper right", "upper left", etc.
            ncol=1,              # typically 1 when inside to avoid taking too much space
            frameon=True,
            framealpha=0.9,
        )

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"resolution_effects_{cfg}.png"
            fig.savefig(fname, dpi=200)
            print(f"Wrote {fname}")
        else:
            plt.show()
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Parse and plot resolution/latency effects.")
    ap.add_argument("infile", type=Path, help="Path to the table text file")
    ap.add_argument("--out_csv", type=Path, default=None, help="Optional CSV output path (tidy format)")
    ap.add_argument("--out_dir", type=Path, default=None, help="Optional directory to write figures")
    args = ap.parse_args()

    text = args.infile.read_text(encoding="utf-8")
    df = parse_table(text)

    print("\nParsed data:")
    print(df.to_string(index=False))

    # Quick pivot for verification
    piv = df.pivot(index=["cfg","lat_ms"], columns="res", values="success_rate").sort_index()
    print("\nPivot (success_rate) rows=(cfg,lat_ms) cols=res:")
    print(piv)

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nWrote CSV: {args.out_csv}")

    plot_per_cfg(df, args.out_dir)


if __name__ == "__main__":
    main()