"""One size, three gravity cases, with shared interaction-energy evaluations.

All energies and trap classifications come from the existing sweep adapter and
strict barrier analysis. The public size and separation coordinates use nm;
native energies remain per area, per length, or per object as appropriate.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

import numpy as np

from .constants import ETHANOL_DENSITY, GOLD_DENSITY
from .material_sets import DEFAULT_COATING_MODEL, DEFAULT_MATERIAL_SET
from .sweeps.adapter import _gravity_energy, evaluate_energy_curve
from .sweeps.analysis import analyze_single_point, reporting_conversion
from .sweeps.models import (
    AdaptiveExtensionConfig, ConvergenceConfig, EnergyCurveResult,
    ExtremaRefinementConfig, NumericalResolutionConfig, SelectionConfig,
    SinglePointResult, SweepPointInput, TrapAnalysisConfig,
)


GRAVITY_CASES = ("gravity_zero", "gravity_positive", "gravity_negative")
_GEOMETRIES = {"plane": "plate", "cylinder": "cylinder", "sphere": "sphere"}
_ENERGY_UNITS = {"plate": "J/m^2", "cylinder": "J/m", "sphere": "J"}
_DEFAULTS = {
    "physics": {
        "temperature_k": 300.0, "teflon_thickness_nm": 60.0,
        "object_density_kg_m3": GOLD_DENSITY, "medium_density_kg_m3": ETHANOL_DENSITY,
        "gravity_m_s2": 9.81,
    },
    "separation": {
        "minimum_nm": 1.0, "maximum_nm": 1000.0, "points": 81,
        "max_extensions": 3, "extension_factor": 2.0,
    },
    "numerics": {
        "plane": {"fs": "psd", "frequency_order": None, "epsrel": 1e-6, "epsabs": 0.0},
        "cylinder": {"Nx": None, "Ny": 40, "nmax": None, "fs": "psd", "frequency_order": None, "epsrel": 1e-6},
        "sphere": {"n_radial": None, "m_angular": None, "psd_order": None, "epsrel": 1e-6, "lmax_factor": None, "min_lmax": None, "cores": 1},
    },
    "analysis": {"refine_extrema": True, "max_refine_evaluations": 32, "grid_convergence": False},
}


def _number(value: Any, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result) or (result <= minimum if strict else result < minimum):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} {minimum}.")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _settings(settings: Mapping[str, Any], geometry: str) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be a mapping.")
    result = deepcopy(_DEFAULTS)
    for section in ("physics", "separation", "analysis"):
        supplied = settings.get(section, {})
        if not isinstance(supplied, Mapping):
            raise ValueError(f"settings.{section} must be a mapping.")
        result[section].update(supplied)
    numerics = settings.get("numerics", {})
    if not isinstance(numerics, Mapping) or not isinstance(numerics.get(geometry, {}), Mapping):
        raise ValueError(f"settings.numerics.{geometry} must be a mapping.")
    result["numerics"][geometry].update(numerics.get(geometry, {}))
    physics, separation, analysis = (result[key] for key in ("physics", "separation", "analysis"))
    for name in _DEFAULTS["physics"]:
        physics[name] = _number(physics[name], f"physics.{name}", strict=name in {"temperature_k", "object_density_kg_m3"})
    for name in ("minimum_nm", "maximum_nm"):
        separation[name] = _number(separation[name], f"separation.{name}", strict=True)
    if separation["maximum_nm"] <= separation["minimum_nm"]:
        raise ValueError("separation.maximum_nm must exceed separation.minimum_nm.")
    separation["points"] = _integer(separation["points"], "separation.points", 5)
    separation["max_extensions"] = _integer(separation["max_extensions"], "separation.max_extensions", 0)
    separation["extension_factor"] = _number(separation["extension_factor"], "separation.extension_factor", minimum=1.0, strict=True)
    for name in ("refine_extrema", "grid_convergence"):
        if not isinstance(analysis[name], bool):
            raise ValueError(f"analysis.{name} must be a boolean.")
    analysis["max_refine_evaluations"] = _integer(analysis["max_refine_evaluations"], "analysis.max_refine_evaluations", 1)
    selected = result["numerics"][geometry]
    if geometry == "cylinder" and selected.get("backend", "exact") not in {"pfa", "exact"}:
        raise ValueError("numerics.cylinder.backend must be pfa or exact.")
    for name in ("Nx", "Ny", "nmax", "frequency_order", "n_radial", "m_angular", "psd_order", "min_lmax", "cores"):
        if selected.get(name) is not None:
            selected[name] = _integer(selected[name], f"numerics.{geometry}.{name}", 1)
    if geometry == "cylinder" and selected["Nx"] is not None and selected["Nx"] % 2:
        raise ValueError("numerics.cylinder.Nx must be even.")
    if "fs" in selected and selected["fs"] not in {"psd", "msd"}:
        raise ValueError(f"numerics.{geometry}.fs must be psd or msd.")
    if selected.get("fs", "psd") == "psd":
        order = selected.get("frequency_order", selected.get("psd_order"))
        if order is not None and order < 2:
            raise ValueError("The PSD frequency order must be at least 2.")
    for name in ("epsrel", "epsabs", "lmax_factor"):
        if selected.get(name) is not None:
            selected[name] = _number(selected[name], f"numerics.{geometry}.{name}", strict=name != "epsabs")
    return result


def _analysis_settings(settings: dict[str, Any]) -> TrapAnalysisConfig:
    separation, analysis = settings["separation"], settings["analysis"]
    return TrapAnalysisConfig(
        profile="coarse", coordinate="log_separation", initial_grid_points=separation["points"],
        adaptive_extension=AdaptiveExtensionConfig(
            enabled=separation["max_extensions"] > 0,
            extension_factor=separation["extension_factor"], max_extensions=separation["max_extensions"],
            edge_fraction=0.10, tail_points=8,
        ),
        extrema_refinement=ExtremaRefinementConfig(
            enabled=analysis["refine_extrema"], method="bounded_backend", xatol=3e-4,
            max_evaluations=analysis["max_refine_evaluations"], merge_distance_rtol=1e-4,
        ),
        numerical_resolution=NumericalResolutionConfig(
            energy_comparison_rtol=1e-7, slope_zero_rtol=1e-6, tail_flatness_rtol=1e-3,
            tail_extension_rtol=1e-3, gravity_dominance_ratio=4.0, energy_atol_native=0.0,
        ),
        selection=SelectionConfig(primary_rule="global_minimum", tie_break_rule="largest_trap_depth_then_smaller_d"),
        convergence=ConvergenceConfig(
            enabled=analysis["grid_convergence"], refined_grid_factor=2,
            d_trap_rtol=5e-3, delta_u_rtol=1e-2,
            require_same_stable_trap_count=True, require_same_selected_branch=True,
        ),
    )


class _SharedInteraction:
    """Cache exact separations only within one fixed physical/numerical context."""

    def __init__(self, geometry: str, parameters: dict[str, Any]) -> None:
        self.geometry = geometry
        self.parameters = deepcopy(parameters)
        self.values: dict[str, float] = {}
        self.metadata: dict[str, Any] = {}
        self.calls = 0

    def __call__(self, geometry, distances_m, parameters, gravity_case):
        if geometry != self.geometry or parameters != self.parameters:
            raise ValueError("The shared interaction cache cannot cross physical or numerical settings.")
        distances = np.asarray(distances_m, dtype=float)
        keys = [float(value).hex() for value in distances]
        missing = np.asarray(sorted({float(value) for key, value in zip(keys, distances) if key not in self.values}))
        if missing.size:
            self.calls += 1
            curve = evaluate_energy_curve(geometry, missing, parameters, "gravity_zero")
            if curve.energy_unit != _ENERGY_UNITS[geometry] or not np.array_equal(curve.distances_m, missing):
                raise ValueError("Interaction backend changed native units or requested separations.")
            values = np.asarray(curve.interaction_energy)
            if np.iscomplexobj(values) or values.shape != missing.shape or not np.all(np.isfinite(values)):
                raise ValueError("Interaction backend returned invalid energy values.")
            self.metadata = deepcopy(curve.metadata)
            self.values.update((float(distance).hex(), float(value)) for distance, value in zip(missing, values))
        interaction = np.asarray([self.values[key] for key in keys], dtype=float)
        gravity, gravity_metadata = _gravity_energy(geometry, distances, parameters, gravity_case)
        return EnergyCurveResult(
            geometry=geometry, gravity_case=gravity_case, distances_m=distances,
            interaction_energy=interaction, gravity_energy=gravity, total_energy=interaction + gravity,
            energy_unit=_ENERGY_UNITS[geometry], metadata={**self.metadata, "gravity": gravity_metadata},
        )


def _scenario(result: SinglePointResult) -> dict[str, Any]:
    selected = next((item for item in result.trap_candidates if item.candidate_index == result.selected_trap_index), None)
    minima = [item for item in result.stationary_points if item.kind == "minimum" and item.refinement_success]
    candidate = min(minima, key=lambda item: (item.total_energy_native, item.d_m), default=None)
    return {
        "gravity_case": result.gravity_case, "status": result.status,
        "trap_exists": result.trap_exists, "trap_classification": result.trap_classification,
        "equilibrium_distance_nm": None if result.d_trap_m is None else result.d_trap_m * 1e9,
        "trap_depth_native": result.delta_u_native, "trap_depth_reporting": result.delta_u_reporting,
        "depth_is_lower_bound": bool(selected is not None and selected.delta_u_is_lower_bound),
        "cutoff_sensitive": result.cutoff_sensitive, "delta_u_infinite": result.delta_u_infinite,
        "candidate_distance_nm": None if candidate is None else candidate.d_m * 1e9,
        "publication_ready": result.publication_ready,
        "local_minimum_count": result.local_minimum_count, "stable_trap_count": result.stable_trap_count,
        "energy_unit": result.energy_unit, "reporting_unit": result.reporting_unit,
        "distances_nm": result.distances_m * 1e9,
        "interaction_energy": result.interaction_energy, "gravity_energy": result.gravity_energy,
        "total_energy": result.total_energy,
        "interaction_energy_reporting": result.interaction_energy_reporting,
        "gravity_energy_reporting": result.gravity_energy_reporting, "total_energy_reporting": result.total_energy_reporting,
        "selected_trap_index": result.selected_trap_index,
        "trap_candidates": [item.to_dict() for item in result.trap_candidates],
        "distance_extensions": [item.to_dict() for item in result.distance_extensions],
        "convergence": None if result.convergence is None else result.convergence.to_dict(),
        "metadata": result.convergence_metadata, "warnings": result.warnings,
        "exception_information": result.exception_information,
    }


def _json_safe(value: Any) -> Any:
    """Keep unavailable diagnostics null, including explicit infinite barriers."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def evaluate_size(geometry: str, size_nm: float, settings: dict[str, Any]) -> dict[str, Any]:
    """Analyze Au thickness (plane) or radius (cylinder/sphere) for three loads.

    Positive gravity points toward the substrate; negative gravity points away.
    Buoyancy is included. Initial separation bounds can be extended by the
    strict tail analysis. A local minimum with unresolved escape barriers is
    reported only as ``candidate_distance_nm``, without a trap depth.
    """
    if geometry not in _GEOMETRIES:
        raise ValueError("geometry must be plane, cylinder, or sphere.")
    size = _number(size_nm, "size_nm")
    config = _settings(settings, geometry)
    backend_geometry = _GEOMETRIES[geometry]
    physics, separation = config["physics"], config["separation"]
    _, reporting_unit = reporting_conversion(backend_geometry, physics["temperature_k"])
    common = {
        "geometry": geometry, "size_nm": size, "energy_unit": _ENERGY_UNITS[backend_geometry],
        "reporting_unit": reporting_unit, "material_set": DEFAULT_MATERIAL_SET,
        "coating_model": DEFAULT_COATING_MODEL, "interaction_evaluations": 0,
        "interaction_backend_calls": 0,
    }
    if geometry == "cylinder":
        common["calculation_method"] = config["numerics"][geometry].get("backend", "exact")
    if size == 0.0:
        scenarios = []
        for gravity_case in GRAVITY_CASES:
            scenarios.append({
                "gravity_case": gravity_case, "status": "zero_size", "trap_exists": False,
                "trap_classification": "degenerate_geometry", "equilibrium_distance_nm": None,
                "trap_depth_native": None, "trap_depth_reporting": None, "depth_is_lower_bound": False,
                "cutoff_sensitive": False, "delta_u_infinite": False, "candidate_distance_nm": None,
                "publication_ready": False, "energy_unit": common["energy_unit"], "reporting_unit": reporting_unit,
                "distances_nm": [], "interaction_energy": [], "gravity_energy": [], "total_energy": [],
                "interaction_energy_reporting": [], "gravity_energy_reporting": [], "total_energy_reporting": [],
                "local_minimum_count": 0, "stable_trap_count": 0, "trap_candidates": [],
                "selected_trap_index": None, "distance_extensions": [], "convergence": None, "metadata": {},
                "warnings": ["Zero object size has no trapping object and no unique equilibrium separation."],
            })
        return {**common, "scenarios": scenarios}
    acceleration = physics["gravity_m_s2"]
    gravity_cases = {
        case: {"effective_acceleration_m_s2": value, "load_kind": "buoyancy_corrected_gravity"}
        for case, value in zip(GRAVITY_CASES, (0.0, -acceleration, acceleration))
    }
    thickness_m = physics["teflon_thickness_nm"] * 1e-9
    dimensions = {
        "t_Au_m": size * 1e-9 if geometry == "plane" else None,
        "t_PTFE_m": thickness_m,
        "R_cyl_m": size * 1e-9 if geometry == "cylinder" else None,
        "R_sph_m": size * 1e-9 if geometry == "sphere" else None,
        "temperature_K": physics["temperature_k"],
    }
    parameters = {
        **dimensions, "gravity_cases": gravity_cases, "buoyancy_included": True,
        "object_density_kg_m3": physics["object_density_kg_m3"],
        "medium_density_kg_m3": physics["medium_density_kg_m3"],
        "material_set": DEFAULT_MATERIAL_SET, "coating_model": DEFAULT_COATING_MODEL,
        "numerical_settings": config["numerics"][geometry],
    }
    maximum_m = separation["maximum_nm"] * 1e-9
    distance_settings = {
        "d_min_m": separation["minimum_nm"] * 1e-9, "d_max_floor_m": maximum_m,
        # Keep the requested initial maximum even when the coating exceeds it.
        "t_PTFE_factor": min(1.0, maximum_m / thickness_m) if thickness_m > 0.0 else 1.0,
        "points": {"coarse": separation["points"]},
    }
    digest = hashlib.sha256(json.dumps(_json_safe(config), sort_keys=True, allow_nan=False).encode()).hexdigest()
    cache = _SharedInteraction(backend_geometry, parameters)
    analysis_config = _analysis_settings(config)
    scenarios = []
    for case in GRAVITY_CASES:
        point = SweepPointInput(
            geometry=backend_geometry, gravity_case=case, **dimensions,
            gravity_parameters=gravity_cases, distance_grid_settings=distance_settings,
            model_version="focused-energy-v1", config_hash=digest, physical_parameters=parameters,
        )
        result = analyze_single_point(point, analysis_config, energy_evaluator=cache)
        scenario = _scenario(result)
        if common.get("calculation_method") == "pfa":
            scenario["equilibrium_gap_over_radius"] = (
                None if scenario["equilibrium_distance_nm"] is None
                else scenario["equilibrium_distance_nm"] / size
            )
            scenario["candidate_gap_over_radius"] = (
                None if scenario["candidate_distance_nm"] is None
                else scenario["candidate_distance_nm"] / size
            )
            scenario["publication_ready"] = False
            scenario["warnings"].append(
                "Teacher cylinder PFA requires separation/radius << 1. "
                "Grid convergence does not validate this geometric approximation or distant escape barriers."
            )
        scenarios.append(scenario)
    return _json_safe({
        **common, "interaction_evaluations": len(cache.values), "interaction_backend_calls": cache.calls,
        "material_reference": cache.metadata.get("native_model", {}).get("material_reference"),
        "scenarios": scenarios,
    })


__all__ = ["GRAVITY_CASES", "evaluate_size"]
