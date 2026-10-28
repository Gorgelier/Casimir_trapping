from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ETHANOL_DENSITY, GOLD_DENSITY, NANOMETER
from .gravity_equilibrium import (
    calculate_gravitational_pressure,
    find_stable_equilibrium_distance_with_gravity,
)
from .material_sets import (
    COATING_MODELS, MATERIAL_SETS, DEFAULT_MATERIAL_SET, DEFAULT_COATING_MODEL,
    select_coating, select_gold_and_medium,
)
from .plane import LayeredPlane
from .system import CasimirSystem

FOCUSED_SPACING_POWER = 1.8


@dataclass(frozen=True)
class SweepConfig:
    teflon_thickness_nm: float
    coating_model: str = DEFAULT_COATING_MODEL
    top_plate_thickness_nm: float = 45.0
    distance_min_nm: float = 10.0
    distance_max_nm: float = 500.0
    points: int = 80
    spacing: str = "focused"
    temperature_k: float = 300.0
    material_set: str = DEFAULT_MATERIAL_SET
    gravity_scenario: int = 0
    top_plate_density: float = GOLD_DENSITY
    gap_medium_density: float = ETHANOL_DENSITY
    fs: str = "psd"
    frequency_order: int | None = None
    epsrel: float = 1.0e-8
    epsabs: float = 0.0
    xi_nodes: int | None = None
    k_nodes: int | None = None

    def __post_init__(self) -> None:
        if self.teflon_thickness_nm < 0.0:
            raise ValueError("teflon_thickness_nm must be non-negative.")
        if self.top_plate_thickness_nm < 0.0:
            raise ValueError("top_plate_thickness_nm must be non-negative.")
        if self.distance_min_nm <= 0.0 or self.distance_max_nm <= 0.0:
            raise ValueError("Distance range must be positive.")
        if self.distance_min_nm >= self.distance_max_nm:
            raise ValueError("distance_min_nm must be smaller than distance_max_nm.")
        if self.points < 2:
            raise ValueError("points must be at least 2.")
        if self.spacing not in {"linear", "log", "focused"}:
            raise ValueError("spacing must be linear, log, or focused.")
        if self.gravity_scenario not in {0, 1, 2}:
            raise ValueError("gravity_scenario must be 0, 1, or 2.")
        if self.coating_model not in COATING_MODELS:
            raise ValueError(f"Unsupported coating_model: {self.coating_model}")
        if self.material_set not in MATERIAL_SETS:
            raise ValueError(f"Unsupported material_set: {self.material_set}")
        if self.fs not in {"psd", "msd"}:
            raise ValueError("fs must be psd or msd.")
        if self.epsrel <= 0.0:
            raise ValueError("epsrel must be positive.")
        if self.temperature_k < 0.0:
            raise ValueError("temperature_k must be non-negative.")
        if (
            self.frequency_order is not None
            and self.xi_nodes is not None
            and self.frequency_order != self.xi_nodes
        ):
            raise ValueError(
                "frequency_order and its legacy alias xi_nodes must match when both are set."
            )
        if self.frequency_order is None and self.xi_nodes is not None:
            object.__setattr__(self, "frequency_order", self.xi_nodes)

    @property
    def teflon_thickness_m(self) -> float:
        return self.teflon_thickness_nm * NANOMETER

    @property
    def top_plate_thickness_m(self) -> float:
        return self.top_plate_thickness_nm * NANOMETER

    def distance_grid_m(self) -> np.ndarray:
        if self.spacing == "log":
            return np.geomspace(self.distance_min_nm, self.distance_max_nm, self.points) * NANOMETER
        if self.spacing == "focused":
            frac = np.linspace(0.0, 1.0, self.points) ** FOCUSED_SPACING_POWER
            return (self.distance_min_nm + (self.distance_max_nm - self.distance_min_nm) * frac) * NANOMETER
        return np.linspace(self.distance_min_nm, self.distance_max_nm, self.points) * NANOMETER


@dataclass(frozen=True)
class SweepResult:
    distances_nm: np.ndarray
    energies_j_per_m2: np.ndarray
    pressures_pa: np.ndarray
    minimum_index: int
    minimum_distance_nm: float
    equilibrium_distance_nm: float | None
    zero_crossing_distance_nm: float | None
    min_energy_j_per_m2: float
    temperature_k: float
    gravity_scenario: int
    gravitational_pressure_pa: float
    equilibrium_with_gravity_nm: float | None


@dataclass(frozen=True)
class PressureSweepResult:
    distances_nm: np.ndarray
    pressures_pa: np.ndarray
    zero_crossing_distance_nm: float | None
    temperature_k: float
    gravity_scenario: int
    gravitational_pressure_pa: float
    equilibrium_with_gravity_nm: float | None


def _value_tolerance(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values))) if len(values) else 0.0
    return max(1.0e-18, np.finfo(float).eps * max(1.0, scale) * 32.0)


def _find_zero_crossing_distance_nm(distances_nm: np.ndarray, values: np.ndarray) -> float | None:
    tolerance = _value_tolerance(values)
    for index in range(1, len(values)):
        previous = float(values[index - 1])
        current = float(values[index])
        if abs(previous) <= tolerance:
            return float(distances_nm[index - 1])
        if abs(current) <= tolerance:
            return float(distances_nm[index])
        if previous * current < 0.0:
            weight = abs(previous) / (abs(previous) + abs(current))
            return float(distances_nm[index - 1] + (distances_nm[index] - distances_nm[index - 1]) * weight)
    return None


def _find_stable_pressure_crossing_distance_nm(
    distances_nm: np.ndarray,
    pressures_pa: np.ndarray,
) -> float | None:
    """Return the pressure crossing with dP/dd < 0, i.e. positive to negative."""
    tolerance = _value_tolerance(pressures_pa)
    signs = np.zeros(len(pressures_pa), dtype=int)
    signs[pressures_pa > tolerance] = 1
    signs[pressures_pa < -tolerance] = -1

    nonzero_indices = np.flatnonzero(signs)
    for left_index, right_index in zip(nonzero_indices[:-1], nonzero_indices[1:]):
        if signs[left_index] <= 0 or signs[right_index] >= 0:
            continue
        if right_index == left_index + 1:
            previous = float(pressures_pa[left_index])
            current = float(pressures_pa[right_index])
            weight = abs(previous) / (abs(previous) + abs(current))
            return float(
                distances_nm[left_index]
                + (distances_nm[right_index] - distances_nm[left_index]) * weight
            )
        return float(distances_nm[left_index + 1])
    return None


def _find_stable_equilibrium_distance_nm(
    distances_nm: np.ndarray,
    energies_j_per_m2: np.ndarray,
    minimum_index: int,
) -> float | None:
    if len(energies_j_per_m2) < 3 or minimum_index <= 0 or minimum_index >= len(energies_j_per_m2) - 1:
        return None

    x0, x1, x2 = distances_nm[minimum_index - 1 : minimum_index + 2]
    y0, y1, y2 = energies_j_per_m2[minimum_index - 1 : minimum_index + 2]
    if y0 <= y1 or y2 <= y1:
        return None

    matrix = np.array([[x0**2, x0, 1.0], [x1**2, x1, 1.0], [x2**2, x2, 1.0]])
    try:
        a, b_coeff, _ = np.linalg.solve(matrix, np.array([y0, y1, y2]))
    except np.linalg.LinAlgError:
        return float(x1)
    if a <= 0.0:
        return float(x1)
    x_min = -b_coeff / (2.0 * a)
    return float(x_min) if x0 <= x_min <= x2 else float(x1)


def build_default_system(distance_m: float, config: SweepConfig) -> CasimirSystem:
    coating_permittivity = select_coating(config.coating_model)
    gold_permittivity, medium_permittivity = select_gold_and_medium(config.material_set)
    top_plane = LayeredPlane(
        materials=(gold_permittivity, medium_permittivity),
        thicknesses_m=(config.top_plate_thickness_m,),
    )
    bottom_plane = LayeredPlane(
        materials=(coating_permittivity, gold_permittivity),
        thicknesses_m=(config.teflon_thickness_m,),
    )
    return CasimirSystem(
        temperature_k=config.temperature_k,
        distance_m=distance_m,
        left_plane=top_plane,
        right_plane=bottom_plane,
        medium_permittivity=medium_permittivity,
        fs=config.fs,
        epsrel=config.epsrel,
        epsabs=config.epsabs,
        frequency_order=config.frequency_order,
        xi_nodes=config.xi_nodes,
        k_nodes=config.k_nodes,
    )


def free_energy_per_area(distance_m: float, config: SweepConfig) -> float:
    return build_default_system(distance_m, config).energy()


def pressure_per_area(distance_m: float, config: SweepConfig) -> float:
    return build_default_system(distance_m, config).pressure()


def calculate_casimir_pressure_direct(
    distance_m: float,
    teflon_thickness_m: float,
    top_plate_thickness_m: float,
    temperature_k: float,
    xi_nodes: int | None,
    k_nodes: int | None,
    coating_model: str,
    material_set: str = DEFAULT_MATERIAL_SET,
) -> float:
    config = SweepConfig(
        teflon_thickness_nm=teflon_thickness_m / NANOMETER,
        top_plate_thickness_nm=top_plate_thickness_m / NANOMETER,
        coating_model=coating_model,
        temperature_k=temperature_k,
        material_set=material_set,
        distance_min_nm=1.0,
        distance_max_nm=2.0,
        points=2,
        frequency_order=xi_nodes,
        xi_nodes=xi_nodes,
        k_nodes=k_nodes,
    )
    return pressure_per_area(distance_m, config)


def sweep_free_energy(config: SweepConfig) -> SweepResult:
    distances_m = config.distance_grid_m()
    distances_nm = distances_m / NANOMETER
    energies = np.array([free_energy_per_area(distance_m, config) for distance_m in distances_m], dtype=float)
    pressures = np.array([pressure_per_area(distance_m, config) for distance_m in distances_m], dtype=float)
    minimum_index = int(np.argmin(energies))
    stable_crossing_nm = _find_stable_pressure_crossing_distance_nm(distances_nm, pressures)
    gravitational_pressure_pa = calculate_gravitational_pressure(
        top_plate_thickness_m=config.top_plate_thickness_m,
        gravity_scenario=config.gravity_scenario,
        top_plate_density=config.top_plate_density,
        gap_medium_density=config.gap_medium_density,
    )
    equilibrium_with_gravity = find_stable_equilibrium_distance_with_gravity(
        distances_nm=distances_nm,
        pressures_pa=pressures,
        gravity_scenario=config.gravity_scenario,
        top_plate_thickness_m=config.top_plate_thickness_m,
        top_plate_density=config.top_plate_density,
        gap_medium_density=config.gap_medium_density,
    )
    return SweepResult(
        distances_nm=distances_nm,
        energies_j_per_m2=energies,
        pressures_pa=pressures,
        minimum_index=minimum_index,
        minimum_distance_nm=float(distances_nm[minimum_index]),
        equilibrium_distance_nm=stable_crossing_nm,
        zero_crossing_distance_nm=stable_crossing_nm,
        min_energy_j_per_m2=float(energies[minimum_index]),
        temperature_k=config.temperature_k,
        gravity_scenario=config.gravity_scenario,
        gravitational_pressure_pa=gravitational_pressure_pa,
        equilibrium_with_gravity_nm=equilibrium_with_gravity,
    )


def sweep_pressure_equilibrium(config: SweepConfig) -> PressureSweepResult:
    distances_m = config.distance_grid_m()
    distances_nm = distances_m / NANOMETER
    pressures = np.array([pressure_per_area(distance_m, config) for distance_m in distances_m], dtype=float)
    stable_crossing_nm = _find_stable_pressure_crossing_distance_nm(distances_nm, pressures)
    gravitational_pressure_pa = calculate_gravitational_pressure(
        top_plate_thickness_m=config.top_plate_thickness_m,
        gravity_scenario=config.gravity_scenario,
        top_plate_density=config.top_plate_density,
        gap_medium_density=config.gap_medium_density,
    )
    equilibrium_with_gravity = find_stable_equilibrium_distance_with_gravity(
        distances_nm=distances_nm,
        pressures_pa=pressures,
        gravity_scenario=config.gravity_scenario,
        top_plate_thickness_m=config.top_plate_thickness_m,
        top_plate_density=config.top_plate_density,
        gap_medium_density=config.gap_medium_density,
    )
    return PressureSweepResult(
        distances_nm=distances_nm,
        pressures_pa=pressures,
        zero_crossing_distance_nm=stable_crossing_nm,
        temperature_k=config.temperature_k,
        gravity_scenario=config.gravity_scenario,
        gravitational_pressure_pa=gravitational_pressure_pa,
        equilibrium_with_gravity_nm=equilibrium_with_gravity,
    )


def write_csv(result: SweepResult, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "distance_nm",
                "free_energy_J_per_m2",
                "pressure_Pa",
                "temperature_K",
                "gravity_scenario",
                "gravitational_pressure_Pa",
                "equilibrium_no_gravity_nm",
                "equilibrium_with_gravity_nm",
            ]
        )
        for distance_nm, energy, pressure in zip(result.distances_nm, result.energies_j_per_m2, result.pressures_pa):
            writer.writerow(
                [
                    f"{distance_nm:.8f}",
                    f"{energy:.12e}",
                    f"{pressure:.12e}",
                    f"{result.temperature_k:.2f}",
                    f"{result.gravity_scenario}",
                    f"{result.gravitational_pressure_pa:.12e}",
                    result.equilibrium_distance_nm if result.equilibrium_distance_nm is not None else "NaN",
                    result.equilibrium_with_gravity_nm if result.equilibrium_with_gravity_nm is not None else "NaN",
                ]
            )
    return output


__all__ = [
    "COATING_MODELS",
    "MATERIAL_SETS",
    "PressureSweepResult",
    "SweepConfig",
    "SweepResult",
    "build_default_system",
    "calculate_casimir_pressure_direct",
    "free_energy_per_area",
    "pressure_per_area",
    "sweep_free_energy",
    "sweep_pressure_equilibrium",
    "write_csv",
]
