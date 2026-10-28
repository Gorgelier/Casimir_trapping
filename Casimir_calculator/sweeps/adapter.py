"""Thin, side-effect-free adapter over the repository's three energy models."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import numpy as np

from Casimir_calculator.caspwn_adapter import (
    NumericalSettings,
    caspwn_energy_joule,
    make_materials,
)
from Casimir_calculator.integrator_finite_temp import SweepConfig, free_energy_per_area
from Casimir_calculator.material_sets import DEFAULT_MATERIAL_SET, DEFAULT_COATING_MODEL

from .models import EnergyCurveResult

UNRESOLVED = "UNRESOLVED"

NativeEvaluator: TypeAlias = Callable[[np.ndarray, Mapping[str, Any]], tuple[np.ndarray, str, dict[str, Any]]]
_GRAVITY_ALIASES = {
    "positive": "gravity_positive",
    "zero": "gravity_zero",
    "negative": "gravity_negative",
    "gravity_positive": "gravity_positive",
    "gravity_zero": "gravity_zero",
    "gravity_negative": "gravity_negative",
}


def _finite_parameter(parameters: Mapping[str, Any], name: str, *, positive: bool = False) -> float:
    value = parameters.get(name)
    if value == UNRESOLVED:
        raise ValueError(f"physical_parameters.{name} is UNRESOLVED; supply an approved SI value before evaluation.")
    if isinstance(value, bool):
        raise ValueError(f"physical_parameters.{name}={value!r}; expected a finite SI number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"physical_parameters.{name}={value!r}; expected a finite SI number.") from exc
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"physical_parameters.{name}={value!r}; expected a {qualifier} SI number.")
    return result


def _nonnegative_parameter(parameters: Mapping[str, Any], name: str) -> float:
    result = _finite_parameter(parameters, name)
    if result < 0.0:
        raise ValueError(f"physical_parameters.{name}={result!r}; expected a non-negative SI value.")
    return result


def _numerical(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    value = parameters.get("numerical_settings", {})
    if not isinstance(value, Mapping):
        raise ValueError("physical_parameters.numerical_settings must be a mapping when provided.")
    return value


def _evaluate_plate(distances_m: np.ndarray, parameters: Mapping[str, Any]) -> tuple[np.ndarray, str, dict[str, Any]]:
    t_au_m = _nonnegative_parameter(parameters, "t_Au_m")
    t_ptfe_m = _nonnegative_parameter(parameters, "t_PTFE_m")
    temperature_k = _finite_parameter(parameters, "temperature_K", positive=True)
    numerical = _numerical(parameters)
    config = SweepConfig(
        teflon_thickness_nm=t_ptfe_m / 1.0e-9,
        top_plate_thickness_nm=t_au_m / 1.0e-9,
        temperature_k=temperature_k,
        material_set=str(parameters.get("material_set", DEFAULT_MATERIAL_SET)),
        coating_model=str(parameters.get("coating_model", DEFAULT_COATING_MODEL)),
        fs=str(numerical.get("fs", "psd")),
        frequency_order=numerical.get("frequency_order"),
        epsrel=float(numerical.get("epsrel", 1.0e-8)),
        epsabs=float(numerical.get("epsabs", 0.0)),
    )
    values = np.asarray([free_energy_per_area(float(distance), config) for distance in distances_m], dtype=float)
    return values, "J/m^2", {"entry_point": "Casimir_calculator.integrator_finite_temp.free_energy_per_area"}


def _sphere_settings(parameters: Mapping[str, Any]) -> NumericalSettings:
    numerical = _numerical(parameters)
    return NumericalSettings(
        n_radial=numerical.get("n_radial"),
        m_angular=numerical.get("m_angular"),
        psd_order=numerical.get("psd_order"),
        epsrel=numerical.get("epsrel"),
        lmax_factor=numerical.get("lmax_factor"),
        min_lmax=numerical.get("min_lmax"),
        cores=int(numerical.get("cores", 1)),
    )


def _evaluate_sphere(distances_m: np.ndarray, parameters: Mapping[str, Any]) -> tuple[np.ndarray, str, dict[str, Any]]:
    radius_m = _finite_parameter(parameters, "R_sph_m", positive=True)
    t_ptfe_m = _nonnegative_parameter(parameters, "t_PTFE_m")
    temperature_k = _finite_parameter(parameters, "temperature_K")
    materials = make_materials(
        str(parameters.get("material_set", DEFAULT_MATERIAL_SET)),
        str(parameters.get("coating_model", DEFAULT_COATING_MODEL)),
    )
    settings = _sphere_settings(parameters)
    values = np.asarray(
        [
            caspwn_energy_joule(
                radius_m=radius_m,
                distance_m=float(distance),
                teflon_thickness_m=t_ptfe_m,
                temperature_k=temperature_k,
                materials=materials,
                settings=settings,
            )
            for distance in distances_m
        ],
        dtype=float,
    )
    return values, "J", {"entry_point": "Casimir_calculator.caspwn_adapter.caspwn_energy_joule"}


def _evaluate_cylinder(distances_m: np.ndarray, parameters: Mapping[str, Any]) -> tuple[np.ndarray, str, dict[str, Any]]:
    # Select the requested teacher backend without importing workflow scripts.
    from Casimir_calculator import cylinder_adapter as cylinder_model

    radius_m = _finite_parameter(parameters, "R_cyl_m", positive=True)
    t_ptfe_m = _nonnegative_parameter(parameters, "t_PTFE_m")
    temperature_k = _finite_parameter(parameters, "temperature_K", positive=True)
    material_set = str(
        parameters.get(
            "cylinder_material_set",
            parameters.get("material_set", cylinder_model.REQUIRED_MATERIAL_SET),
        )
    )
    coating_model = str(
        parameters.get(
            "cylinder_coating_model",
            parameters.get("coating_model", cylinder_model.REQUIRED_COATING_MODEL),
        )
    )
    if material_set != cylinder_model.REQUIRED_MATERIAL_SET:
        raise ValueError(
            "Cylinder sweeps require material_set="
            f"{cylinder_model.REQUIRED_MATERIAL_SET!r}; received {material_set!r}."
        )
    if coating_model != cylinder_model.REQUIRED_COATING_MODEL:
        raise ValueError(
            "Cylinder sweeps require coating_model="
            f"{cylinder_model.REQUIRED_COATING_MODEL!r}; received {coating_model!r}."
        )

    configured_roles = {
        "cylinder_material": str(parameters.get("cylinder_material", "gold")),
        "medium_material": str(parameters.get("medium_material", "ethanol")),
        "coating_material": str(parameters.get("cylinder_coating_material", "teflon")),
        "substrate_material": str(parameters.get("cylinder_substrate_material", "gold")),
    }
    for role, required_name in cylinder_model.REQUIRED_ROLE_MATERIALS.items():
        actual_name = configured_roles[role]
        if actual_name != required_name:
            raise ValueError(
                f"Reference-material cylinder sweeps require {role}={required_name!r}; "
                f"received {actual_name!r}."
            )

    cylinder_material, medium_material, coating_material = make_materials(
        material_set,
        coating_model,
    )
    substrate_material = cylinder_material
    if t_ptfe_m == 0.0:
        plane_materials = [substrate_material]
        plane_thicknesses_m = [None]
    else:
        plane_materials = [coating_material, substrate_material]
        plane_thicknesses_m = [t_ptfe_m, None]
    numerical = _numerical(parameters)

    # The configuration selects the backend. The original scattering calculation
    # below runs for backend="exact" and is skipped for backend="pfa".
    # --cylinder-method overrides this selection for a new scan.
    # Legacy configurations without a backend field keep their original method.
    backend = numerical.get("backend", "exact")
    if backend not in {"pfa", "exact"}:
        raise ValueError("numerics.cylinder.backend must be pfa or exact")
    if backend == "pfa":
        from Casimir_calculator.cylinder_pfa_adapter import pfa_energy_per_length, PFA_NUMERICS

        values = [pfa_energy_per_length(
            temperature_k, float(distance), radius_m, cylinder_material,
            plane_materials, plane_thicknesses_m, medium_material,
        ) for distance in distances_m]
        return np.asarray(values, dtype=float), "J/m", {
            "entry_point": "casimir_cylinder.pfa_cypl.compute_pfa",
            "calculation_method": "pfa", "approximation": "proximity force approximation",
            "observable": "energy", "numerics": PFA_NUMERICS,
            "radius_scaling": "interaction energy only: sqrt(R/R_reference)",
            "validity_requirement": "surface separation / cylinder radius << 1",
        }

    values: list[float] = []
    for distance_m in distances_m:
        system = cylinder_model.LayeredCylinderPlaneSystem(
            temperature_k,
            float(distance_m),
            radius_m,
            cylinder_material,
            plane_materials,
            plane_thicknesses_m,
            medium_material,
        )
        values.append(
            system.calculate(
                "energy",
                Nx=numerical.get("Nx"),
                Ny=int(numerical.get("Ny", 40)),
                nmax=numerical.get("nmax"),
                fs=str(numerical.get("fs", "psd")),
                epsrel=float(numerical.get("epsrel", 1.0e-8)),
                N=numerical.get("frequency_order"),
            )
        )
    return np.asarray(values, dtype=float), "J/m", {
        "entry_point": "Casimir_calculator.cylinder_adapter.LayeredCylinderPlaneSystem.calculate",
        "observable": "energy", "calculation_method": "exact",
    }


_INTERACTION_EVALUATORS: dict[str, NativeEvaluator] = {
    "plate": _evaluate_plate,
    "cylinder": _evaluate_cylinder,
    "sphere": _evaluate_sphere,
}


def _normalized_gravity_case(gravity_case: str) -> str:
    try:
        return _GRAVITY_ALIASES[gravity_case]
    except KeyError as exc:
        raise ValueError(
            f"gravity_case={gravity_case!r}; allowed={sorted(_GRAVITY_ALIASES)!r}."
        ) from exc


def _gravity_case_parameters(parameters: Mapping[str, Any], gravity_case: str) -> Mapping[str, Any]:
    cases = parameters.get("gravity_cases")
    if not isinstance(cases, Mapping):
        raise ValueError("physical_parameters.gravity_cases must contain the three configured load cases.")
    case = cases.get(gravity_case)
    if not isinstance(case, Mapping):
        raise ValueError(f"physical_parameters.gravity_cases.{gravity_case} is missing or not a mapping.")
    return case


def _gravity_energy(
    geometry: str,
    distances_m: np.ndarray,
    parameters: Mapping[str, Any],
    gravity_case: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    case = _gravity_case_parameters(parameters, gravity_case)
    acceleration = case.get("effective_acceleration_m_s2")
    if acceleration == UNRESOLVED:
        raise ValueError(
            f"gravity_cases.{gravity_case}.effective_acceleration_m_s2 is UNRESOLVED; "
            "supply an approved signed acceleration before energy evaluation."
        )
    if isinstance(acceleration, bool):
        raise ValueError(f"gravity_cases.{gravity_case}.effective_acceleration_m_s2 must be numeric.")
    try:
        acceleration_value = float(acceleration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"gravity_cases.{gravity_case}.effective_acceleration_m_s2 must be numeric.") from exc
    if not math.isfinite(acceleration_value):
        raise ValueError(f"gravity_cases.{gravity_case}.effective_acceleration_m_s2 must be finite.")
    if acceleration_value == 0.0:
        return np.zeros_like(distances_m), {
            "load_kind": case.get("load_kind"),
            "effective_acceleration_m_s2": 0.0,
            "potential_reference": "U_g(d=0)=0",
        }

    object_density = _finite_parameter(parameters, "object_density_kg_m3", positive=True)
    medium_density = _finite_parameter(parameters, "medium_density_kg_m3")
    include_buoyancy = parameters.get("buoyancy_included")
    if include_buoyancy is not True:
        raise ValueError(
            f"physical_parameters.buoyancy_included={include_buoyancy!r}; "
            "the current models require an explicit True/False decision before nonzero-gravity evaluation."
        )
    density_contrast = object_density - medium_density
    if geometry == "plate":
        measure = _nonnegative_parameter(parameters, "t_Au_m")
        load = density_contrast * measure * acceleration_value
        load_unit = "N/m^2"
    elif geometry == "cylinder":
        radius = _finite_parameter(parameters, "R_cyl_m", positive=True)
        measure = math.pi * radius**2
        load = density_contrast * measure * acceleration_value
        load_unit = "N/m"
    elif geometry == "sphere":
        radius = _finite_parameter(parameters, "R_sph_m", positive=True)
        measure = 4.0 * math.pi * radius**3 / 3.0
        load = density_contrast * measure * acceleration_value
        load_unit = "N"
    else:
        raise NotImplementedError(f"No gravity-energy measure is implemented for geometry {geometry!r}.")
    # Existing force convention is positive away from the substrate and
    # F=-dU/dd, so the constant-load potential is -F*d. The d=0 reference is
    # explicit metadata and does not affect barriers or extrema.
    return -load * distances_m, {
        "load_kind": case.get("load_kind"),
        "signed_load": load,
        "load_unit": load_unit,
        "effective_acceleration_m_s2": acceleration_value,
        "potential_reference": "U_g(d=0)=0",
        "buoyancy_included": True,
    }


def evaluate_energy_curve(
    geometry: str,
    distances_m: np.ndarray,
    physical_parameters: Mapping[str, Any],
    gravity_case: str,
) -> EnergyCurveResult:
    """Evaluate native interaction energy plus the configured gravity potential.

    No trap classification, plotting, or file writing occurs here. The adapter
    preserves the native geometry-specific SI energy unit.
    """
    distances = np.asarray(distances_m, dtype=float)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("distances_m must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(distances)) or np.any(distances <= 0.0):
        raise ValueError("distances_m must contain only positive finite SI values.")
    if distances.size > 1 and np.any(np.diff(distances) <= 0.0):
        raise ValueError("distances_m must be strictly increasing.")
    try:
        evaluator = _INTERACTION_EVALUATORS[geometry]
    except KeyError as exc:
        raise NotImplementedError(
            f"No existing energy model is registered for geometry {geometry!r}; "
            f"available geometries are {sorted(_INTERACTION_EVALUATORS)}."
        ) from exc
    normalized_case = _normalized_gravity_case(gravity_case)
    interaction, unit, native_metadata = evaluator(distances, physical_parameters)
    interaction = np.asarray(interaction, dtype=float)
    if interaction.shape != distances.shape:
        raise ValueError(
            f"{geometry} energy backend returned shape {interaction.shape}; expected {distances.shape}."
        )
    if not np.all(np.isfinite(interaction)):
        raise FloatingPointError(f"{geometry} energy backend returned NaN or infinity.")
    gravity, gravity_metadata = _gravity_energy(
        geometry,
        distances,
        physical_parameters,
        normalized_case,
    )
    total = interaction + gravity
    selected_set = physical_parameters.get("material_set", DEFAULT_MATERIAL_SET)
    selected_coating = physical_parameters.get("coating_model", DEFAULT_COATING_MODEL)
    if geometry == "cylinder":
        selected_set = physical_parameters.get("cylinder_material_set", selected_set)
        selected_coating = physical_parameters.get("cylinder_coating_model", selected_coating)
    if selected_set == DEFAULT_MATERIAL_SET or selected_coating == DEFAULT_COATING_MODEL:
        from Casimir_calculator.reference_materials import cylinder_reference_provenance
        native_metadata = {
            **native_metadata,
            "material_set": selected_set,
            "coating_model": selected_coating,
            "material_reference": cylinder_reference_provenance(
                material_set=selected_set, coating_model=selected_coating,
            ),
        }
    return EnergyCurveResult(
        geometry=geometry,
        gravity_case=normalized_case,
        distances_m=distances,
        interaction_energy=interaction,
        gravity_energy=gravity,
        total_energy=total,
        energy_unit=unit,
        metadata={"native_model": native_metadata, "gravity": gravity_metadata},
    )


__all__ = ["EnergyCurveResult", "evaluate_energy_curve"]
