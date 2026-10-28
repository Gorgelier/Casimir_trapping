"""Export two figures per geometry without inventing missing equilibria or depths."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


GRAVITY_CASES = ("gravity_zero", "gravity_positive", "gravity_negative")
GRAVITY_STYLES = {
    "gravity_zero": ("No gravity", "#2563eb", "o", "-"),
    "gravity_positive": ("Positive gravity (toward substrate)", "#d94830", "s", "--"),
    "gravity_negative": ("Negative gravity (away from substrate)", "#16815d", "D", "-."),
}
GEOMETRIES = {
    "plane": ("plane_thickness", "Plane–plane", "Gold plate thickness (nm)", r"$k_B T/\mu\mathrm{m}^2$"),
    "cylinder": ("cylinder_radius", "Cylinder–plane", "Cylinder radius (nm)", r"$k_B T/\mu\mathrm{m}$"),
    "sphere": ("sphere_radius", "Sphere–plane", "Sphere radius (nm)", r"$k_B T$"),
}
SUMMARY_FIELDS = (
    "geometry", "size_nm", "gravity_case", "status", "equilibrium_distance_nm",
    "candidate_distance_nm", "trap_depth_native", "trap_depth_reporting", "energy_unit",
    "reporting_unit", "depth_is_lower_bound", "cutoff_sensitive", "delta_u_infinite",
    "interaction_evaluations", "plot_value", "plot_kind",
    "calculation_method", "equilibrium_gap_over_radius", "candidate_gap_over_radius",
)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _scheduled_results(results: list[dict], geometry: str, settings: dict | None) -> list[dict]:
    """Include missing scheduled sizes so a failed/pending point leaves a gap."""
    selected = [item for item in results if item["geometry"] == geometry]
    spec = (settings or {}).get("scans", {}).get(geometry, {})
    if not selected or not all(key in spec for key in ("minimum_nm", "maximum_nm")):
        return selected
    if "step_nm" not in spec and spec.get("spacing") != "log":
        return selected
    # The saved schedule and figures must share one grid implementation. In
    # particular, a logarithmic schedule includes an explicit zero plus positive
    # radii; reconstructing it as a linear step would silently bridge missing points.
    from .focused_scan import size_grid

    sizes = size_grid(spec)
    existing = {float(item["size_nm"]).hex() for item in selected}
    template = selected[0]
    missing = [{
        "geometry": geometry, "size_nm": size,
        "energy_unit": template.get("energy_unit"), "reporting_unit": template.get("reporting_unit"),
        "interaction_evaluations": None,
        "scenarios": [{"gravity_case": case, "status": "not_evaluated"} for case in GRAVITY_CASES],
    } for size in sizes if float(size).hex() not in existing]
    # Explicit additional points remain available, including diagnostic exports.
    return [*selected, *missing]


def _records(results: list[dict], geometry: str, settings: dict | None = None) -> list[dict]:
    records: list[dict] = []
    seen = set()
    for result in sorted(_scheduled_results(results, geometry, settings), key=lambda item: float(item["size_nm"])):
        size = float(result["size_nm"])
        if not math.isfinite(size) or size < 0.0:
            raise ValueError("size_nm must be finite and non-negative")
        scenarios = {item["gravity_case"]: item for item in result["scenarios"]}
        if len(scenarios) != len(result["scenarios"]) or not set(scenarios).issubset(GRAVITY_CASES):
            raise ValueError("Each size must have unique, recognized gravity cases")
        for gravity_case in GRAVITY_CASES:
            key = (size, gravity_case)
            if key in seen:
                raise ValueError(f"Duplicate result for {geometry}, {size:g} nm, {gravity_case}")
            seen.add(key)
            scenario = scenarios.get(gravity_case, {"status": "not_evaluated"})
            record = {**scenario, "geometry": geometry, "size_nm": size, "gravity_case": gravity_case}
            for key in ("energy_unit", "reporting_unit", "interaction_evaluations", "calculation_method"):
                record[key] = result.get(key)
            records.append(record)
    return records


def _plot_value(record: dict, quantity: str) -> tuple[float | None, str]:
    # A vanishing object has no trapping equilibrium or trap depth, even if a
    # caller accidentally includes a numerical placeholder for it.
    if record["size_nm"] == 0.0:
        return None, "degenerate_zero_size"
    if quantity == "energy_equilibrium_distances":
        value = _finite(record.get("equilibrium_distance_nm"))
        if value is not None:
            return value, "equilibrium"
        candidate = _finite(record.get("candidate_distance_nm"))
        return (candidate, "unresolved_candidate") if candidate is not None else (None, "missing")
    if record.get("delta_u_infinite"):
        return None, "infinite_depth"
    value = _finite(record.get("trap_depth_reporting"))
    if value is None:
        return None, "missing"
    if record.get("depth_is_lower_bound"):
        return value, "lower_bound"
    if record.get("status") not in {"success", "multiple_traps"}:
        return None, "unresolved_depth"
    return value, "depth"


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _make_figure(records: list[dict], geometry: str, quantity: str, settings: dict):
    _, geometry_title, xlabel, depth_unit = GEOMETRIES[geometry]
    figure, axis = plt.subplots(figsize=(8.6, 5.5))
    is_distance = quantity == "energy_equilibrium_distances"
    title = "Stable equilibrium distance" if is_distance else "Trap depth"
    is_pfa = geometry == "cylinder" and settings.get("numerics", {}).get("cylinder", {}).get("backend", "exact") == "pfa"
    if is_pfa:
        title += " (teacher PFA)"
    figure.suptitle(f"{geometry_title}: {title}", x=0.13, y=0.97, ha="left", fontsize=16, fontweight="bold")
    axis.set_xlabel(xlabel, fontsize=11)
    axis.set_ylabel("Equilibrium separation (nm)" if is_distance else f"Trap depth ({depth_unit})", fontsize=11)
    bounds = settings.get("scans", {}).get(geometry, {})
    minimum = float(bounds.get("minimum_nm", 10.0))
    maximum = float(bounds.get("maximum_nm", 1000.0))
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
        raise ValueError(f"Invalid plotting size range for {geometry}")
    log_size_axis = bounds.get("spacing") == "log"
    if log_size_axis:
        positive_minimum = minimum if minimum > 0.0 else float(bounds.get("positive_minimum_nm", 0.0))
        if not math.isfinite(positive_minimum) or not 0.0 < positive_minimum < maximum:
            raise ValueError(f"Invalid positive logarithmic plotting range for {geometry}")
        axis.set_xscale("log")
        axis.set_xlim(positive_minimum, maximum)
    else:
        axis.set_xlim(minimum, maximum)
    primary = "equilibrium" if is_distance else "depth"
    special_kinds: set[str] = set()
    finite_count = 0
    for gravity_case in GRAVITY_CASES:
        label, color, marker, linestyle = GRAVITY_STYLES[gravity_case]
        rows = [record for record in records if record["gravity_case"] == gravity_case]
        if log_size_axis:
            # Zero is a degenerate case retained in the table, not a logarithmic
            # coordinate. Positive missing points still remain as NaN gaps.
            rows = [record for record in rows if record["size_nm"] > 0.0]
        values = [_plot_value(record, quantity) for record in rows]
        x = [record["size_nm"] for record in rows]
        y = [value if kind == primary else np.nan for value, kind in values]
        # Keep NaN entries in the plotted vector: gaps must not be bridged.
        axis.plot(x, y, color=color, marker=marker, linestyle=linestyle, linewidth=1.7, markersize=4.5, label=label)
        for record, (value, kind) in zip(rows, values, strict=True):
            if value is None:
                continue
            finite_count += 1
            if kind in {"unresolved_candidate", "lower_bound"}:
                axis.scatter(record["size_nm"], value, marker="o" if kind == "unresolved_candidate" else "^",
                             s=45, facecolors="none", edgecolors=color, linewidths=1.5, zorder=4)
                special_kinds.add(kind)
    handles, labels = axis.get_legend_handles_labels()
    for kind, marker, label in (
        ("unresolved_candidate", "o", "Unresolved candidate (not a confirmed equilibrium)"),
        ("lower_bound", "^", "Lower bound: depth is at least the marked value"),
    ):
        if kind in special_kinds:
            handles.append(Line2D([], [], color="#64748b", marker=marker, markerfacecolor="none", linestyle="none"))
            labels.append(label)
    axis.legend(handles, labels, loc="best", frameon=False, fontsize=8.4)
    if finite_count == 0:
        axis.text(0.5, 0.49, "No resolved finite equilibrium in this scan" if is_distance else "No resolved finite trap depth in this scan",
                  transform=axis.transAxes, ha="center", va="center", color="#64748b", fontsize=11)
    axis.grid(axis="both", color="#e2e8f0", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=10)
    notes = ["Gaps indicate missing or unresolved values."]
    if is_pfa:
        notes.append("PFA approximation: requires separation/radius << 1; not validated by grid convergence.")
    if log_size_axis:
        notes.append("Logarithmic size axis: positive sizes only; zero size is retained in the CSV.")
    if any(record["size_nm"] == 0.0 for record in records):
        notes.append("Zero size: no physical object; equilibrium and depth are undefined.")
    if any(record.get("delta_u_infinite") for record in records) and not is_distance:
        notes.append("Infinite depths are retained in the CSV and omitted from the finite axis.")
    figure.text(0.13, 0.035, "\n".join(notes), fontsize=8, color="#64748b", va="bottom")
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.20, top=0.88)
    return figure, axis


def export_results(results: list[dict], output_dir: Path, settings: dict) -> list[Path]:
    """Write two PNG/SVG figures, two summary CSVs and raw curves per geometry."""
    unknown = {item.get("geometry") for item in results} - GEOMETRIES.keys()
    if unknown:
        raise ValueError(f"Unsupported geometry: {unknown}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for geometry, (prefix, _, _, _) in GEOMETRIES.items():
        records = _records(results, geometry, settings)
        if not records:
            continue
        for quantity in ("energy_equilibrium_distances", "trap_depths"):
            csv_path = output_dir / f"{prefix}_{quantity}.csv"
            summary_rows = []
            for record in records:
                value, kind = _plot_value(record, quantity)
                summary_rows.append({**record, "plot_value": value, "plot_kind": kind})
            _write_csv(csv_path, SUMMARY_FIELDS, summary_rows)
            paths.append(csv_path)
            figure, _ = _make_figure(records, geometry, quantity, settings)
            try:
                for extension in ("png", "svg"):
                    path = output_dir / f"{prefix}_{quantity}.{extension}"
                    figure.savefig(path, dpi=180, metadata={"Date": None} if extension == "svg" else None)
                    paths.append(path)
            finally:
                plt.close(figure)
        curve_rows = []
        for record in records:
            arrays = [record.get(key, []) for key in ("distances_nm", "interaction_energy", "gravity_energy", "total_energy")]
            if len({len(array) for array in arrays}) > 1:
                raise ValueError("Energy curve arrays must have equal lengths")
            for distance, interaction, gravity, total in sorted(zip(*arrays, strict=True), key=lambda row: float(row[0])):
                curve_rows.append({"geometry": geometry, "size_nm": record["size_nm"], "gravity_case": record["gravity_case"],
                                   "status": record.get("status"), "distance_nm": distance, "energy_unit": record.get("energy_unit"),
                                   "interaction_energy_native": interaction, "gravity_energy_native": gravity, "total_energy_native": total})
        curve_path = output_dir / f"{prefix}_energy_curves.csv"
        _write_csv(curve_path, ("geometry", "size_nm", "gravity_case", "status", "distance_nm", "energy_unit",
                               "interaction_energy_native", "gravity_energy_native", "total_energy_native"), curve_rows)
        paths.append(curve_path)
    return paths
