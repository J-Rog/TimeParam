import pandas as pd
import numpy as np
import re
from pathlib import Path

def aggregate_logs_from_filenames(base_dir: str = ".", routes=(0, 1)):
    """
    Aggregate route simulation logs. Resolution and route_id are inferred from filenames.

    Filenames must follow the pattern: '<resolution>r_route<id>.csv'
    e.g., '0.75r_route0.csv', '1.0r_route1.csv'
    """
    #(?:_resaware)?
    pattern = re.compile(r"(?P<resolution>[0-9.]+)r_resaware_route(?P<route_id>\d+)\.csv$")
    dfs = []

    for p in Path(base_dir).rglob("*.csv"):
        m = pattern.search(p.name)
        if not m:
            continue
        route_id = int(m.group("route_id"))
        if route_id not in routes:
            continue
        resolution = float(m.group("resolution"))
        df = pd.read_csv(p)
        df["resolution"] = resolution
        df["route_id"] = f"RouteScenario_{route_id}_rep0"
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No matching files for pattern <resolution>r_route<id>.csv")

    df = pd.concat(dfs, ignore_index=True)

    # Convert numerics
    for col, default in {
        "control_latency": 0.0,
        "latency_steps": 0
    }.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = [
        "control_period","control_latency",
        "control_steps","latency_steps",
        "vehicle_density","pedestrian_density","route_completion",
        "sim_time","wall_time","lane_crossings",
        "collisions_vehicle","collisions_pedestrian","collisions_static","red_lights"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["timestamp"] = pd.to_datetime(df.get("timestamp", pd.NaT), errors="coerce")
    df["control_period_rounded"] = df["control_period"].round(2)
    df["control_latency_rounded"] = df["control_latency"].round(2)
    df["success"] = (df["status"] == "completed").astype(int)

    collision_cols = ["collisions_vehicle","collisions_pedestrian","collisions_static"]
    for col in collision_cols:
        if col not in df.columns:
            df[col] = 0

    df["any_collision"] = (
        df[collision_cols]
        .fillna(0)
        .gt(0)
        .any(axis=1)
        .astype(int)
    )
    df["vehicle_collision_flag"] = (df["collisions_vehicle"] > 0).astype(int)
    df["ped_collision_flag"] = (df["collisions_pedestrian"] > 0).astype(int)
    df["static_collision_flag"] = (df["collisions_static"] > 0).astype(int)

    # ---- 1. Summary by resolution ----
    agg_by_res = (
        df.groupby("resolution")
        .agg(
            runs=("status","count"),
            success_rate=("success","mean"),
            collision_rate=("any_collision","mean"),
            vehicle_collision_rate=("vehicle_collision_flag","mean"),
            ped_collision_rate=("ped_collision_flag","mean"),
            static_collision_rate=("static_collision_flag","mean"),
            mean_completion=("route_completion","mean"),
            mean_sim_time=("sim_time","mean"),
            mean_wall_time=("wall_time","mean"),
            mean_lane_crossings=("lane_crossings","mean"),
            sum_collisions_vehicle=("collisions_vehicle","sum"),
            sum_collisions_ped=("collisions_pedestrian","sum"),
            sum_collisions_static=("collisions_static","sum"),
            sum_red_lights=("red_lights","sum")
        ).reset_index()
    )
    rate_cols = [
        "success_rate",
        "collision_rate",
        "vehicle_collision_rate",
        "ped_collision_rate",
        "static_collision_rate"
    ]
    for col in rate_cols:
        agg_by_res[col] = (100 * agg_by_res[col]).round(1)

    # ---- 2. Success & safety by resolution × control period ----
    agg_res_cp = (
        df.groupby(["resolution","control_period_rounded","control_latency_rounded"])
        .agg(
            runs=("status","count"),
            success_rate=("success","mean"),
            collision_rate=("any_collision","mean"),
            vehicle_collision_rate=("vehicle_collision_flag","mean"),
            ped_collision_rate=("ped_collision_flag","mean"),
            static_collision_rate=("static_collision_flag","mean"),
            mean_completion=("route_completion","mean"),
            median_lane_crossings=("lane_crossings","median"),
            mean_lane_crossings=("lane_crossings","mean"),
            std_lane_crossings=("lane_crossings","std"),
            sum_collisions_vehicle=("collisions_vehicle","sum"),
            sum_collisions_ped=("collisions_pedestrian","sum"),
            sum_collisions_static=("collisions_static","sum"),
            red_lights=("red_lights","sum")
        ).reset_index()
        .sort_values(["resolution","control_period_rounded","control_latency_rounded"])
    )
    rate_cols_cp = [
        "success_rate",
        "collision_rate",
        "vehicle_collision_rate",
        "ped_collision_rate",
        "static_collision_rate"
    ]
    for col in rate_cols_cp:
        agg_res_cp[col] = (100 * agg_res_cp[col]).round(1)
    agg_res_cp["mean_completion"] = agg_res_cp["mean_completion"].round(2)
    agg_res_cp["median_lane_crossings"] = agg_res_cp["median_lane_crossings"].round(2)
    agg_res_cp["mean_lane_crossings"] = agg_res_cp["mean_lane_crossings"].round(2)
    agg_res_cp["std_lane_crossings"] = (
        agg_res_cp["std_lane_crossings"]
        .fillna(0)
        .round(2)
    )

    # ---- 3. Environment density effects ----
    density_group_cols = ["vehicle_density","pedestrian_density"]
    if "resolution" in df.columns:
        density_group_cols.append("resolution")

    agg_density = (
        df.groupby(density_group_cols)
        .agg(
            runs=("status","count"),
            success_rate=("success","mean"),
            collision_rate=("any_collision","mean"),
            vehicle_collision_rate=("vehicle_collision_flag","mean"),
            ped_collision_rate=("ped_collision_flag","mean"),
            static_collision_rate=("static_collision_flag","mean"),
            mean_completion=("route_completion","mean"),
            mean_lane_crossings=("lane_crossings","mean"),
            median_lane_crossings=("lane_crossings","median")
        )
        .reset_index()
        .sort_values(density_group_cols)
    )
    density_rate_cols = [
        "success_rate",
        "collision_rate",
        "vehicle_collision_rate",
        "ped_collision_rate",
        "static_collision_rate"
    ]
    for col in density_rate_cols:
        agg_density[col] = (100 * agg_density[col]).round(1)
    agg_density["mean_completion"] = agg_density["mean_completion"].round(2)
    agg_density["mean_lane_crossings"] = agg_density["mean_lane_crossings"].round(2)
    agg_density["median_lane_crossings"] = agg_density["median_lane_crossings"].round(2)

    # ---- 4. Failures ----
    failures = (
        df[df["status"] != "completed"]
        .groupby([
            "resolution",
            "control_period_rounded",
            "control_latency_rounded",
            "vehicle_density",
            "pedestrian_density",
            "message"
        ])
        .size()
        .reset_index(name="count")
    )

    # ---- 5. By route ----
    agg_by_route = (
        df.groupby(["resolution","route_id","control_period_rounded","control_latency_rounded"])
        .agg(
            runs=("status","count"),
            success_rate=("success","mean"),
            collision_rate=("any_collision","mean"),
            mean_lane_crossings=("lane_crossings","mean")
        ).reset_index()
        .sort_values(["resolution","route_id","control_period_rounded","control_latency_rounded"])
    )
    agg_by_route["success_rate"] = (100 * agg_by_route["success_rate"]).round(1)
    agg_by_route["collision_rate"] = (100 * agg_by_route["collision_rate"]).round(1)
    agg_by_route["mean_lane_crossings"] = agg_by_route["mean_lane_crossings"].round(2)

    # Print summaries
    print("\n=== Summary by resolution ===")
    print(agg_by_res.to_string(index=False))

    print("\n=== Success and safety by resolution × control period × control latency ===")
    print(agg_res_cp.to_string(index=False))

    print("\n=== Effects of environment density ===")
    print(agg_density.to_string(index=False))

    print("\n=== Failures (non-completions) ===")
    print(failures.to_string(index=False))

    print("\n=== By route scenario ===")
    print(agg_by_route.to_string(index=False))

    # ---- Save CSVs for Google Sheets ----
    out_dir = Path(base_dir) / "aggregated_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_by_res.to_csv(out_dir / "summary_by_resolution.csv", index=False)
    agg_res_cp.to_csv(out_dir / "success_by_resolution_control.csv", index=False)
    agg_density.to_csv(out_dir / "by_density.csv", index=False)
    failures.to_csv(out_dir / "failures.csv", index=False)
    agg_by_route.to_csv(out_dir / "by_route.csv", index=False)

    print(f"\nCSV outputs written to: {out_dir.resolve()}")

    return {
        "by_resolution": agg_by_res,
        "by_resolution_cp": agg_res_cp,
        "by_density": agg_density,
        "failures": failures,
        "by_route": agg_by_route
    }


# Example usage:
results = aggregate_logs_from_filenames("", routes=(0, 1))
