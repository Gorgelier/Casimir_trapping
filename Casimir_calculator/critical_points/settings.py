"""Outward-load critical loci on the fixed PTFE grid 10, 15, ..., 200 nm."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

THICKNESSES = list(range(10, 201, 5))
DEFAULTS = {
    "physics": {"temperature_K": 300.0, "gold_density_kg_m3": 19320.0,
                "ethanol_density_kg_m3": 785.06, "gravity_m_s2": 9.81,
                "material_set": "cylinder-reference", "coating_model": "teflon-cylinder-reference"},
    "bounds_nm": {"thickness": [10.0, 200.0], "size": [10.0, 1000.0], "gap": [0.5, 100000.0]},
    "solver": {"max_iterations": 18, "max_evaluations": 160, "max_seconds": 2400.0,
               "force_relative_tolerance": 1e-6, "gradient_scaled_tolerance": 1e-5,
               "stationarity_tolerance_nm": 0.005, "max_log_step": 0.18,
               "jacobian_refresh_iterations": 6},
    "profiles": {
        "plane": {"fs": "psd", "frequency_order": 48, "epsrel": 1e-9},
        "cylinder": {"zero": [192, 192, 32, 256], "nonzero": [40, 40, 64, 256], "epsrel": 1e-8},
        "sphere": {"N": 56, "M": 52, "lmax": 900, "O": 48, "fs": "psd", "epsrel": 1e-8, "cores": 1}},
    "sweep": {"thicknesses_nm": THICKNESSES},
    "seeds": {
        "plane": [{"thickness_nm": 150.0, "size_nm": 127.472341, "gap_nm": 165.161838, "branch": "upper_size"}],
        "cylinder": [{"thickness_nm": 90.0, "size_nm": 442.099644, "gap_nm": 78.274323, "branch": "upper_size"},
                     {"thickness_nm": 150.0, "size_nm": 74.105771, "gap_nm": 169.085724, "branch": "upper_size"},
                     {"thickness_nm": 150.0, "size_nm": 12.1242, "gap_nm": 308.606, "branch": "lower_size"}],
        "sphere": [{"thickness_nm": 60.0, "size_nm": 789.016703, "gap_nm": 41.176326, "branch": "upper_size"},
                   {"thickness_nm": 90.0, "size_nm": 363.249908, "gap_nm": 69.407416, "branch": "upper_size"}]},
    "seed_note": "Historical coordinates are initial guesses only; every reported converged row is recomputed with the selected backend."
}


def canonical_geometry(value):
    value = "plane" if value == "plate" else value
    if value not in ("plane", "cylinder", "sphere"):
        raise ValueError("geometry must be plane (or plate), cylinder, or sphere")
    return value


def load_settings(path=None):
    result = copy.deepcopy(DEFAULTS)
    if path is not None:
        updates = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        def merge(a, b):
            for key, value in b.items():
                if isinstance(value, dict) and isinstance(a.get(key), dict):
                    merge(a[key], value)
                else:
                    a[key] = value
        merge(result, updates)
    for name, limits in result["bounds_nm"].items():
        if len(limits) != 2 or not all(math.isfinite(x) and x > 0 for x in limits) or limits[0] >= limits[1]:
            raise ValueError(f"Invalid positive finite bounds for {name}")
    for name, value in result["solver"].items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid solver setting {name}")
    for name in ("max_iterations", "max_evaluations", "jacobian_refresh_iterations"):
        value = result["solver"][name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a positive integer")
    physics = result["physics"]
    for name in ("temperature_K", "gold_density_kg_m3", "ethanol_density_kg_m3", "gravity_m_s2"):
        if not math.isfinite(physics[name]) or physics[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if physics["gold_density_kg_m3"] <= physics["ethanol_density_kg_m3"]:
        raise ValueError("The configured outward load requires gold density greater than medium density")
    for name, values in [("thickness", result["sweep"]["thicknesses_nm"]),
                         *(("size", values) for values in result["sweep"].get("fixed_sizes_nm", {}).values())]:
        lo, hi = result["bounds_nm"][name]
        if any(not math.isfinite(v) or not lo <= v <= hi for v in values):
            raise ValueError(f"Sweep {name} coordinates lie outside configured bounds")
    return result
