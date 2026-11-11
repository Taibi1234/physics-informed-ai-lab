r"""Export validated experiment artifacts for the static GitHub Pages site.

Run from the repository root after installing the project dependencies:
    python scripts\export_website_data.py

The script uses only the Python standard library. It reads the preserved CSV
artifacts and writes docs/data/experiment_data.json without changing results.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADVANCED_DIR = PROJECT_ROOT / "reports" / "advanced_damped_oscillator"
WEIGHT_SUMMARY = (
    PROJECT_ROOT / "reports" / "weight_sweep" / "pinn_weight_summary.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "docs" / "data" / "experiment_data.json"

NUMERIC_RUN_FIELDS = (
    "position_rmse",
    "position_mae",
    "physics_residual_rmse",
    "final_total_loss",
)
NUMERIC_SUMMARY_FIELDS = (
    "runs",
    "position_rmse_mean",
    "position_rmse_std",
    "position_mae_mean",
    "physics_residual_rmse_mean",
    "final_total_loss_mean",
)
NUMERIC_WEIGHT_FIELDS = (
    "seeds",
    "position_mean",
    "position_std",
    "position_median",
    "residual_mean",
    "residual_std",
    "residual_median",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a non-empty CSV and reject missing or duplicate headers."""
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment artifact is missing: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if not headers:
            raise ValueError(f"CSV has no header: {path}")
        if len(headers) != len(set(headers)):
            raise ValueError(f"CSV has duplicate headers: {path}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def finite_number(value: str, field: str, source: Path) -> float:
    """Convert a CSV value to a finite float with a useful error message."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid numeric value for {field!r} in {source}: {value!r}"
        ) from error
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {field!r} in {source}")
    return number


def require_columns(
    rows: list[dict[str, str]], required: set[str], source: Path
) -> None:
    """Fail early when an artifact does not match the expected schema."""
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {source}: {sorted(missing)}")


def export_runs(rows: list[dict[str, str]], source: Path) -> list[dict[str, Any]]:
    require_columns(rows, {"seed", "mode", *NUMERIC_RUN_FIELDS}, source)
    exported: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for row in rows:
        seed = int(finite_number(row["seed"], "seed", source))
        mode = row["mode"].strip()
        key = (seed, mode)
        if key in seen:
            raise ValueError(f"Duplicate run for seed={seed}, mode={mode}")
        seen.add(key)
        record: dict[str, Any] = {"seed": seed, "mode": mode}
        for field in NUMERIC_RUN_FIELDS:
            record[field] = finite_number(row[field], field, source)
        exported.append(record)

    return exported


def export_summary(
    rows: list[dict[str, str]], source: Path
) -> list[dict[str, Any]]:
    require_columns(rows, {"mode", *NUMERIC_SUMMARY_FIELDS}, source)
    exported: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {"mode": row["mode"].strip()}
        for field in NUMERIC_SUMMARY_FIELDS:
            value = finite_number(row[field], field, source)
            record[field] = int(value) if field == "runs" else value
        exported.append(record)
    return exported


def export_predictions(
    rows: list[dict[str, str]], source: Path
) -> list[dict[str, Any]]:
    """Group dense trajectories into compact [time, exact, predicted] arrays."""
    require_columns(
        rows,
        {"seed", "mode", "time", "exact_position", "predicted_position"},
        source,
    )
    grouped: dict[tuple[int, str], list[list[float]]] = {}

    for row in rows:
        seed = int(finite_number(row["seed"], "seed", source))
        mode = row["mode"].strip()
        values = [
            finite_number(row[field], field, source)
            for field in ("time", "exact_position", "predicted_position")
        ]
        grouped.setdefault((seed, mode), []).append(values)

    series = []
    for (seed, mode), points in sorted(grouped.items()):
        points.sort(key=lambda point: point[0])
        if len(points) < 2:
            raise ValueError(f"Trajectory is too short for seed={seed}, mode={mode}")
        if any(points[index][0] >= points[index + 1][0] for index in range(len(points) - 1)):
            raise ValueError(f"Trajectory time is not increasing for seed={seed}, mode={mode}")
        series.append(
            {
                "seed": seed,
                "mode": mode,
                # Seven decimal places preserve the plotted trajectories while
                # avoiding unnecessary JSON size from full binary floats.
                "points": [[round(value, 7) for value in point] for point in points],
            }
        )
    return series


def export_weight_sweep(
    rows: list[dict[str, str]], source: Path
) -> list[dict[str, Any]]:
    require_columns(
        rows,
        {"physics_weight", *NUMERIC_WEIGHT_FIELDS, "std_note"},
        source,
    )
    exported: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "physics_weight": finite_number(
                row["physics_weight"], "physics_weight", source
            ),
            "std_note": row["std_note"].strip(),
        }
        for field in NUMERIC_WEIGHT_FIELDS:
            value = finite_number(row[field], field, source)
            record[field] = int(value) if field == "seeds" else value
        exported.append(record)
    return sorted(exported, key=lambda record: record["physics_weight"])


def main() -> None:
    run_path = ADVANCED_DIR / "damped_results.csv"
    summary_path = ADVANCED_DIR / "damped_summary.csv"
    prediction_path = ADVANCED_DIR / "damped_predictions.csv"
    config_path = ADVANCED_DIR / "damped_config.json"

    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)

    runs = export_runs(read_csv(run_path), run_path)
    summary = export_summary(read_csv(summary_path), summary_path)
    trajectories = export_predictions(read_csv(prediction_path), prediction_path)
    weight_sweep = export_weight_sweep(read_csv(WEIGHT_SUMMARY), WEIGHT_SUMMARY)

    run_keys = {(run["seed"], run["mode"]) for run in runs}
    trajectory_keys = {(series["seed"], series["mode"]) for series in trajectories}
    if run_keys != trajectory_keys:
        raise ValueError("Run metrics and dense trajectories do not cover the same runs")

    payload = {
        "schema_version": 1,
        "project_version": "1.0.0",
        "study": "advanced_damped_oscillator",
        "config": config,
        "runs": runs,
        "summary": summary,
        "trajectories": trajectories,
        "undamped_weight_sweep": weight_sweep,
        "sources": {
            "runs": "reports/advanced_damped_oscillator/damped_results.csv",
            "summary": "reports/advanced_damped_oscillator/damped_summary.csv",
            "trajectories": (
                "reports/advanced_damped_oscillator/damped_predictions.csv"
            ),
            "config": "reports/advanced_damped_oscillator/damped_config.json",
            "weight_sweep": "reports/weight_sweep/pinn_weight_summary.csv",
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")

    print(
        f"Exported {len(runs)} runs, {len(trajectories)} trajectories, "
        f"and {len(weight_sweep)} weight settings to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
