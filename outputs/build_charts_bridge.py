#!/usr/bin/env python3
"""
Bridge raw sweep logs into chart tables and figures.

This script replaces the manual workflow of:
1) running aggregate with different route subsets/model files
2) moving numbers to Google Sheets for calculations
3) preparing chart*.txt manually

It reads sweep CSV logs directly from `outputs/`, computes the chart metrics,
writes `chart1.txt`..`chart4.txt`, and renders figures via existing scripts:
  - outputs/gen_chart1.py
  - outputs/gen_chart2.py
  - outputs/gen_chart3+4.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeriesSpec:
    cfg: str
    res: float
    file_globs: tuple[str, ...]


SERIES_SPECS: tuple[SeriesSpec, ...] = (
    SeriesSpec("NRA", 0.75, ("0.75r_route*.csv",)),
    SeriesSpec("NRA", 1.0, ("1.0r_route*.csv", "1.00r_route*.csv")),
    SeriesSpec("RA", 0.75, ("0.75r_resaware_route*.csv",)),
    SeriesSpec("RA", 1.0, ("1.0r_resaware_route*.csv", "1.00r_resaware_route*.csv")),
)


DEFAULT_ROUTE_GROUPS = ("0,1", "2,3")
DEFAULT_CHART12_LAT_MS = (0, 50)
DEFAULT_CHART34_LAT_MS = (0, 50, 100, 150, 200)
ROUTE_FILE_RE = re.compile(r"route(?P<route>\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build chart data and figures from sweep CSV logs.")
    ap.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing per-route sweep CSV logs.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / "chart_bridge",
        help="Directory to write chart text/CSV/PNG outputs.",
    )
    ap.add_argument(
        "--route-groups",
        nargs="+",
        default=list(DEFAULT_ROUTE_GROUPS),
        help='Route groups to combine (default: "0,1" "2,3").',
    )
    ap.add_argument(
        "--chart12-latencies-ms",
        nargs="+",
        type=int,
        default=list(DEFAULT_CHART12_LAT_MS),
        help="Latencies (ms) used by chart1 and chart2.",
    )
    ap.add_argument(
        "--chart34-latencies-ms",
        nargs="+",
        type=int,
        default=list(DEFAULT_CHART34_LAT_MS),
        help="Latencies (ms) used by chart3 and chart4.",
    )
    ap.add_argument(
        "--std-ddof",
        type=int,
        default=1,
        help="ddof used for lane-crossing stddev (Sheets STDEV.S ~= 1).",
    )
    ap.add_argument(
        "--round-group-rates",
        type=int,
        default=1,
        help=(
            "Round each route-group success/collision rate to this many decimals "
            "before averaging groups. Use 1 to mirror aggregate.py output style."
        ),
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing points/files instead of failing.",
    )
    ap.add_argument(
        "--no-render",
        action="store_true",
        help="Only write chart text/CSV files, do not run plotting scripts.",
    )
    return ap.parse_args()


def parse_route_groups(raw_groups: Sequence[str]) -> list[tuple[int, ...]]:
    groups: list[tuple[int, ...]] = []
    for raw in raw_groups:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"Invalid empty route group: {raw!r}")
        group = tuple(sorted({int(p) for p in parts}))
        groups.append(group)
    if not groups:
        raise ValueError("At least one route group is required.")
    return groups


def route_id_from_filename(path: Path) -> int | None:
    m = ROUTE_FILE_RE.search(path.name)
    if not m:
        return None
    return int(m.group("route"))


def pick_latest_file_per_route(paths: Iterable[Path]) -> dict[int, Path]:
    selected: dict[int, Path] = {}
    for p in paths:
        route_id = route_id_from_filename(p)
        if route_id is None:
            continue
        cur = selected.get(route_id)
        if cur is None or p.stat().st_mtime > cur.stat().st_mtime:
            selected[route_id] = p
    return selected


def load_series_frames(
    logs_dir: Path,
    route_union: set[int],
    allow_missing: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for spec in SERIES_SPECS:
        matched: list[Path] = []
        for g in spec.file_globs:
            matched.extend(logs_dir.glob(g))
        by_route = pick_latest_file_per_route(matched)

        missing_routes = sorted(route_union - set(by_route.keys()))
        if missing_routes and not allow_missing:
            globs = ", ".join(spec.file_globs)
            raise FileNotFoundError(
                f"Missing files for {spec.cfg} {spec.res} routes {missing_routes}. "
                f"Searched globs: {globs} in {logs_dir}"
            )

        for route_id, path in sorted(by_route.items()):
            if route_id not in route_union:
                continue
            df = pd.read_csv(path)
            df["cfg"] = spec.cfg
            df["res"] = float(spec.res)
            df["route_group_id"] = route_id
            df["source_file"] = path.name
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No matching sweep logs found in {logs_dir}")
    out = pd.concat(frames, ignore_index=True)
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "control_latency" not in out.columns:
        if "control_period" in out.columns:
            out["control_latency"] = out["control_period"]
        else:
            out["control_latency"] = np.nan

    numeric_cols = [
        "control_latency",
        "lane_crossings",
        "red_lights",
        "collisions_vehicle",
        "collisions_pedestrian",
        "collisions_static",
    ]
    for c in numeric_cols:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce")

    status = out.get("status", "").astype(str).str.lower().str.strip()
    out["success"] = (status == "completed").astype(int)
    out["any_collision"] = (
        out[["collisions_vehicle", "collisions_pedestrian", "collisions_static"]]
        .fillna(0)
        .gt(0)
        .any(axis=1)
        .astype(int)
    )
    out["lat_ms"] = (out["control_latency"] * 1000).round().astype("Int64")
    return out


def point_subset(
    df: pd.DataFrame,
    cfg: str,
    res: float,
    lat_ms: int,
    routes: Sequence[int],
) -> pd.DataFrame:
    return df[
        (df["cfg"] == cfg)
        & (np.isclose(df["res"], res))
        & (df["lat_ms"] == lat_ms)
        & (df["route_group_id"].isin(routes))
    ]


def safe_std(values: pd.Series, ddof: int) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    if len(vals) <= ddof:
        return 0.0
    return float(vals.std(ddof=ddof))


def format_res(res: float) -> str:
    if np.isclose(res, 1.0):
        return "1.0"
    return f"{res:.2f}".rstrip("0").rstrip(".")


def ensure_not_nan(value: float, label: str) -> None:
    if pd.isna(value):
        raise ValueError(f"Missing value for {label}")


def build_chart_tables(
    df: pd.DataFrame,
    route_groups: Sequence[tuple[int, ...]],
    chart12_latencies_ms: Sequence[int],
    chart34_latencies_ms: Sequence[int],
    std_ddof: int,
    round_group_rates: int,
    allow_missing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chart1_rows: list[dict] = []
    chart2_rows: list[dict] = []
    chart34_rows: list[dict] = []
    coverage_rows: list[dict] = []

    cfg_order = ("NRA", "RA")
    res_order = (0.75, 1.0)

    for lat_ms in chart12_latencies_ms:
        for cfg in cfg_order:
            for res in res_order:
                group_infractions: list[float] = []
                group_means: list[float] = []
                group_stds: list[float] = []
                total_runs = 0

                for i, rg in enumerate(route_groups):
                    sub = point_subset(df, cfg, res, lat_ms, rg)
                    runs = len(sub)
                    total_runs += runs
                    coverage_rows.append(
                        {
                            "chart": "chart1_2",
                            "cfg": cfg,
                            "res": res,
                            "lat_ms": lat_ms,
                            "route_group": ",".join(map(str, rg)),
                            "group_idx": i,
                            "runs": runs,
                        }
                    )
                    if runs == 0:
                        continue

                    infractions = float(sub["red_lights"].fillna(0).sum())
                    group_infractions.append(infractions)

                    lane_vals = pd.to_numeric(sub["lane_crossings"], errors="coerce")
                    group_means.append(float(lane_vals.mean()))
                    group_stds.append(safe_std(lane_vals, ddof=std_ddof))

                if not group_infractions:
                    if allow_missing:
                        continue
                    raise ValueError(f"No data for {cfg} res={res} lat={lat_ms}ms (chart1/2)")

                infractions_total = int(round(sum(group_infractions)))
                mean_avg = float(np.nanmean(group_means))
                std_avg = float(np.nanmean(group_stds))

                chart1_rows.append(
                    {
                        "cfg": cfg,
                        "lat_ms": lat_ms,
                        "res": res,
                        "infractions": infractions_total,
                        "runs": total_runs,
                    }
                )
                chart2_rows.append(
                    {
                        "cfg": cfg,
                        "lat_ms": lat_ms,
                        "res": res,
                        "mean": mean_avg,
                        "sd": std_avg,
                        "runs": total_runs,
                    }
                )

    for cfg in cfg_order:
        for res in res_order:
            for lat_ms in chart34_latencies_ms:
                group_success: list[float] = []
                group_collision: list[float] = []
                total_runs = 0

                for i, rg in enumerate(route_groups):
                    sub = point_subset(df, cfg, res, lat_ms, rg)
                    runs = len(sub)
                    total_runs += runs
                    coverage_rows.append(
                        {
                            "chart": "chart3_4",
                            "cfg": cfg,
                            "res": res,
                            "lat_ms": lat_ms,
                            "route_group": ",".join(map(str, rg)),
                            "group_idx": i,
                            "runs": runs,
                        }
                    )
                    if runs == 0:
                        continue
                    s_rate = round(float(100.0 * sub["success"].mean()), round_group_rates)
                    c_rate = round(float(100.0 * sub["any_collision"].mean()), round_group_rates)
                    group_success.append(s_rate)
                    group_collision.append(c_rate)

                if not group_success:
                    if allow_missing:
                        continue
                    raise ValueError(f"No data for {cfg} res={res} lat={lat_ms}ms (chart3/4)")

                chart34_rows.append(
                    {
                        "cfg": cfg,
                        "res": res,
                        "lat_ms": lat_ms,
                        "success_rate": float(np.mean(group_success)),
                        "collision_rate": float(np.mean(group_collision)),
                        "runs": total_runs,
                    }
                )

    chart1 = pd.DataFrame(chart1_rows).sort_values(["lat_ms", "cfg", "res"]).reset_index(drop=True)
    chart2 = pd.DataFrame(chart2_rows).sort_values(["lat_ms", "cfg", "res"]).reset_index(drop=True)
    chart34 = pd.DataFrame(chart34_rows).sort_values(["cfg", "res", "lat_ms"]).reset_index(drop=True)
    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["chart", "cfg", "res", "lat_ms", "group_idx"]
    ).reset_index(drop=True)
    return chart1, chart2, chart34, coverage


def write_chart1_txt(df: pd.DataFrame, path: Path) -> None:
    lines = ["table for light infractions"]
    for _, row in df.iterrows():
        ensure_not_nan(row["infractions"], f"chart1 {row['cfg']} {row['res']} {row['lat_ms']}ms")
        lines.append(
            f"{row['cfg']} {int(row['lat_ms'])}ms {format_res(float(row['res']))} {int(row['infractions'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chart2_txt(df: pd.DataFrame, path: Path) -> None:
    lines = ["table for lane stats mean std"]
    for _, row in df.iterrows():
        ensure_not_nan(row["mean"], f"chart2 mean {row['cfg']} {row['res']} {row['lat_ms']}ms")
        ensure_not_nan(row["sd"], f"chart2 sd {row['cfg']} {row['res']} {row['lat_ms']}ms")
        lines.append(
            f"{row['cfg']} {int(row['lat_ms'])}ms {format_res(float(row['res']))} "
            f"{float(row['mean']):.12g} {float(row['sd']):.12g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_resolution_effects_txt(df: pd.DataFrame, cfg: str, path: Path) -> None:
    sub = df[df["cfg"] == cfg].sort_values(["res", "lat_ms"])
    if sub.empty:
        raise ValueError(f"No rows for cfg={cfg} when writing {path}")
    lines = ["table for resolution effects success rate collision rate"]
    for res in sorted(sub["res"].unique()):
        lines.append(f"{cfg} {format_res(float(res))}")
        cur = sub[sub["res"] == res].sort_values("lat_ms")
        for _, row in cur.iterrows():
            ensure_not_nan(
                row["success_rate"],
                f"{cfg} success {row['res']} {row['lat_ms']}ms",
            )
            ensure_not_nan(
                row["collision_rate"],
                f"{cfg} collision {row['res']} {row['lat_ms']}ms",
            )
            lines.append(
                f"{int(row['lat_ms'])} ms {float(row['success_rate']):.2f} "
                f"{float(row['collision_rate']):.2f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def render_charts(
    script_dir: Path,
    out_dir: Path,
    render_chart1: bool,
    render_chart2: bool,
    render_chart3: bool,
    render_chart4: bool,
) -> None:
    # Remove stale files from prior runs so reported PNGs reflect this run.
    for stale in (
        "chart1.png",
        "chart2.png",
        "chart3.png",
        "chart4.png",
        "resolution_effects_NRA.png",
        "resolution_effects_RA.png",
    ):
        p = out_dir / stale
        if p.exists():
            p.unlink()

    chart1_txt = out_dir / "chart1.txt"
    chart2_txt = out_dir / "chart2.txt"
    chart3_txt = out_dir / "chart3.txt"
    chart4_txt = out_dir / "chart4.txt"

    if render_chart1 and chart1_txt.exists():
        run_cmd(
            [
                sys.executable,
                str(script_dir / "gen_chart1.py"),
                str(chart1_txt),
                "--out_csv",
                str(out_dir / "chart1_pivot.csv"),
                "--out_png",
                str(out_dir / "chart1.png"),
            ]
        )
    if render_chart2 and chart2_txt.exists():
        run_cmd(
            [
                sys.executable,
                str(script_dir / "gen_chart2.py"),
                str(chart2_txt),
                "--out_csv",
                str(out_dir / "chart2_long.csv"),
                "--out_png",
                str(out_dir / "chart2.png"),
            ]
        )

    # Run once per file so output names map directly to chart3/chart4.
    if render_chart3 and chart3_txt.exists():
        run_cmd(
            [
                sys.executable,
                str(script_dir / "gen_chart3+4.py"),
                str(chart3_txt),
                "--out_csv",
                str(out_dir / "chart3_long.csv"),
                "--out_dir",
                str(out_dir),
            ]
        )
    if render_chart4 and chart4_txt.exists():
        run_cmd(
            [
                sys.executable,
                str(script_dir / "gen_chart3+4.py"),
                str(chart4_txt),
                "--out_csv",
                str(out_dir / "chart4_long.csv"),
                "--out_dir",
                str(out_dir),
            ]
        )

    nra_png = out_dir / "resolution_effects_NRA.png"
    ra_png = out_dir / "resolution_effects_RA.png"
    if nra_png.exists():
        shutil.copy2(nra_png, out_dir / "chart3.png")
    if ra_png.exists():
        shutil.copy2(ra_png, out_dir / "chart4.png")


def main() -> None:
    args = parse_args()
    route_groups = parse_route_groups(args.route_groups)
    route_union = {r for group in route_groups for r in group}

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_series_frames(
        logs_dir=args.logs_dir,
        route_union=route_union,
        allow_missing=args.allow_missing,
    )
    data = normalize_columns(raw)

    chart1_df, chart2_df, chart34_df, coverage_df = build_chart_tables(
        df=data,
        route_groups=route_groups,
        chart12_latencies_ms=args.chart12_latencies_ms,
        chart34_latencies_ms=args.chart34_latencies_ms,
        std_ddof=args.std_ddof,
        round_group_rates=args.round_group_rates,
        allow_missing=args.allow_missing,
    )

    chart1_df.to_csv(out_dir / "chart1_data.csv", index=False)
    chart2_df.to_csv(out_dir / "chart2_data.csv", index=False)
    chart34_df.to_csv(out_dir / "chart34_data.csv", index=False)
    coverage_df.to_csv(out_dir / "coverage.csv", index=False)

    generated_text_files: list[Path] = []
    if not chart1_df.empty:
        p = out_dir / "chart1.txt"
        write_chart1_txt(chart1_df, p)
        generated_text_files.append(p)
    if not chart2_df.empty:
        p = out_dir / "chart2.txt"
        write_chart2_txt(chart2_df, p)
        generated_text_files.append(p)
    if not chart34_df[chart34_df["cfg"] == "NRA"].empty:
        p = out_dir / "chart3.txt"
        write_resolution_effects_txt(chart34_df, "NRA", p)
        generated_text_files.append(p)
    if not chart34_df[chart34_df["cfg"] == "RA"].empty:
        p = out_dir / "chart4.txt"
        write_resolution_effects_txt(chart34_df, "RA", p)
        generated_text_files.append(p)

    expected_chart12_points = {
        (cfg, float(res), int(lat))
        for cfg in ("NRA", "RA")
        for res in (0.75, 1.0)
        for lat in args.chart12_latencies_ms
    }
    actual_chart12_points = {
        (str(r.cfg), float(r.res), int(r.lat_ms))
        for r in chart1_df.itertuples(index=False)
    }
    can_render_chart1_2 = expected_chart12_points.issubset(actual_chart12_points)
    can_render_chart3 = not chart34_df[chart34_df["cfg"] == "NRA"].empty
    can_render_chart4 = not chart34_df[chart34_df["cfg"] == "RA"].empty

    render_ok = False
    if not args.no_render:
        try:
            render_charts(
                script_dir=Path(__file__).resolve().parent,
                out_dir=out_dir,
                render_chart1=can_render_chart1_2,
                render_chart2=can_render_chart1_2,
                render_chart3=can_render_chart3,
                render_chart4=can_render_chart4,
            )
            render_ok = True
        except subprocess.CalledProcessError:
            if not args.allow_missing:
                raise
            print("Rendering skipped for one or more charts due to partial/missing data.")

    print(f"Wrote outputs to: {out_dir.resolve()}")
    print("Generated files:")
    for p in generated_text_files:
        print(f"  {p.name}")
    print("  chart1_data.csv, chart2_data.csv, chart34_data.csv, coverage.csv")
    if render_ok:
        pngs = [
            n
            for n in ("chart1.png", "chart2.png", "chart3.png", "chart4.png")
            if (out_dir / n).exists()
        ]
        if pngs:
            print(f"  {', '.join(pngs)}")
        else:
            print("  (No PNGs generated)")
    elif not args.no_render:
        print("  (PNG rendering incomplete due to missing/partial data)")


if __name__ == "__main__":
    main()
