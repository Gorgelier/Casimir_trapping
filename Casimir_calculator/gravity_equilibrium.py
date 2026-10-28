from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .constants import ETHANOL_DENSITY, G_ACCEL, GOLD_DENSITY

CrossingPredicate = Callable[[float, float], bool]


def calculate_gravitational_pressure(
        top_plate_thickness_m: float,
        gravity_scenario: int = 0,
        top_plate_density: float = GOLD_DENSITY,
        gap_medium_density: float = ETHANOL_DENSITY,
) -> float:
    """Return the buoyancy-corrected gravitational pressure for a finite plate.

    The magnitude is

        P_g = (rho_plate - rho_medium) * thickness * g.

    Sign convention:
        gravity_scenario = 0: no gravity term.
        gravity_scenario = 1: gravity pulls toward the substrate, so P_g < 0.
        gravity_scenario = 2: gravity points away from the substrate, so P_g > 0.
    """
    if gravity_scenario == 0:
        return 0.0
    if gravity_scenario not in {1, 2}:
        raise ValueError("gravity_scenario must be 0, 1, or 2")
    if top_plate_thickness_m < 0.0:
        raise ValueError("top_plate_thickness_m must be non-negative.")
    if top_plate_density <= 0.0 or gap_medium_density <= 0.0:
        raise ValueError("Densities must be positive.")

    pressure_magnitude = (
        (top_plate_density - gap_medium_density)
        * top_plate_thickness_m
        * G_ACCEL
    )

    return -pressure_magnitude if gravity_scenario == 1 else pressure_magnitude


def _validate_sample_arrays(
        distances_nm: np.ndarray,
        values: np.ndarray,
        value_name: str = "values",
) -> None:
    """Validate sampled one-dimensional arrays before zero-crossing interpolation."""
    if distances_nm.ndim != 1 or values.ndim != 1:
        raise ValueError("distances_nm and values must be one-dimensional arrays.")
    if len(distances_nm) != len(values):
        raise ValueError(f"distances_nm and {value_name} must have the same length.")
    if len(distances_nm) < 2:
        raise ValueError("At least two sample points are required.")
    if not np.all(np.isfinite(distances_nm)):
        raise ValueError("distances_nm must contain only finite values.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{value_name} must contain only finite values.")
    if np.any(np.diff(distances_nm) <= 0.0):
        raise ValueError("distances_nm must be strictly increasing.")


def _value_tolerance(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values))) if len(values) else 0.0
    return max(1.0e-18, np.finfo(float).eps * max(1.0, scale) * 32.0)


def _interpolate_crossing_distance_nm(
        distances_nm: np.ndarray,
        values: np.ndarray,
        is_crossing: CrossingPredicate,
        *,
        require_bracket_around_zeros: bool = False,
) -> float | None:
    """Linearly interpolate the first sampled crossing that satisfies a predicate."""
    _validate_sample_arrays(distances_nm, values)

    tolerance = _value_tolerance(values)

    index = 0
    while index < len(values):
        current = float(values[index])

        if abs(current) <= tolerance:
            first_zero = index
            while index + 1 < len(values) and abs(float(values[index + 1])) <= tolerance:
                index += 1
            last_zero = index

            zero_is_accepted = not require_bracket_around_zeros
            if require_bracket_around_zeros:
                left_index = first_zero - 1
                right_index = last_zero + 1
                zero_is_accepted = (
                    left_index >= 0
                    and right_index < len(values)
                    and is_crossing(
                        float(values[left_index]),
                        float(values[right_index]),
                    )
                )

            if zero_is_accepted:
                return float(
                    0.5
                    * (distances_nm[first_zero] + distances_nm[last_zero])
                )

            index += 1
            continue

        next_index = index + 1
        if next_index < len(values):
            following = float(values[next_index])
            if abs(following) > tolerance and is_crossing(current, following):
                weight = abs(current) / (abs(current) + abs(following))
                return float(
                    distances_nm[index]
                    + (distances_nm[next_index] - distances_nm[index]) * weight
                )

        index += 1

    return None


def find_zero_crossing_distance_with_gravity(
        distances_nm: np.ndarray,
        pressures_pa: np.ndarray,
        gravity_scenario: int = 0,
        top_plate_thickness_m: float = 0.0,
        top_plate_density: float = GOLD_DENSITY,
        gap_medium_density: float = ETHANOL_DENSITY,
) -> float | None:
    """Find any zero of the net pressure.

    The solved equation is

        P_net(d) = P_Casimir(d) + P_g = 0.
    """
    _validate_sample_arrays(distances_nm, pressures_pa, value_name="pressures_pa")

    gravitational_pressure = calculate_gravitational_pressure(
        top_plate_thickness_m=top_plate_thickness_m,
        gravity_scenario=gravity_scenario,
        top_plate_density=top_plate_density,
        gap_medium_density=gap_medium_density,
    )

    net_pressures = pressures_pa + gravitational_pressure

    return _interpolate_crossing_distance_nm(
        distances_nm,
        net_pressures,
        is_crossing=lambda previous, current: previous * current < 0.0,
    )


def find_stable_equilibrium_distance_with_gravity(
        distances_nm: np.ndarray,
        pressures_pa: np.ndarray,
        gravity_scenario: int = 0,
        top_plate_thickness_m: float = 0.0,
        top_plate_density: float = GOLD_DENSITY,
        gap_medium_density: float = ETHANOL_DENSITY,
) -> float | None:
    """Find a stable net-pressure equilibrium distance.

    Pressure sign convention:
        P_Casimir > 0 is repulsive and increases the gap.
        P_Casimir < 0 is attractive and decreases the gap.

    Stability condition:
        P_net(d) = P_Casimir(d) + P_g = 0
        dP_net/dd < 0

    Therefore, in an increasing distance grid, a stable sampled crossing is
    positive-to-negative.
    """
    _validate_sample_arrays(distances_nm, pressures_pa, value_name="pressures_pa")

    gravitational_pressure = calculate_gravitational_pressure(
        top_plate_thickness_m=top_plate_thickness_m,
        gravity_scenario=gravity_scenario,
        top_plate_density=top_plate_density,
        gap_medium_density=gap_medium_density,
    )

    net_pressures = pressures_pa + gravitational_pressure

    return _interpolate_crossing_distance_nm(
        distances_nm,
        net_pressures,
        is_crossing=lambda previous, current: previous > 0.0 and current < 0.0,
        require_bracket_around_zeros=True,
    )


def calculate_equilibrium_vs_thickness(
        teflon_thickness_nm: float,
        thickness_range_nm: np.ndarray,
        gravity_scenarios: list[int] | None = None,
        config_kwargs: dict | None = None,
) -> dict:
    """Scan top-plate thickness and return equilibrium distances by gravity scenario."""
    from .integrator_finite_temp import SweepConfig, sweep_pressure_equilibrium

    if gravity_scenarios is None:
        gravity_scenarios = [0, 1, 2]
    if config_kwargs is None:
        config_kwargs = {}

    if thickness_range_nm.ndim != 1:
        raise ValueError("thickness_range_nm must be a one-dimensional array.")
    if not np.all(np.isfinite(thickness_range_nm)):
        raise ValueError("thickness_range_nm must contain only finite values.")
    if np.any(thickness_range_nm < 0.0):
        raise ValueError("thickness_range_nm must be non-negative.")

    results = {
        "thickness_nm": thickness_range_nm.tolist(),
        "no_gravity": [],
        "positive_gravity": [],
        "negative_gravity": [],
    }

    result_keys = {
        0: "no_gravity",
        1: "positive_gravity",
        2: "negative_gravity",
    }

    for scenario in gravity_scenarios:
        if scenario not in result_keys:
            raise ValueError("gravity_scenarios can only contain 0, 1, or 2.")

    for thickness_nm in thickness_range_nm:
        for scenario in gravity_scenarios:
            config = SweepConfig(
                teflon_thickness_nm=teflon_thickness_nm,
                top_plate_thickness_nm=thickness_nm,
                gravity_scenario=scenario,
                **config_kwargs,
            )
            result = sweep_pressure_equilibrium(config)
            results[result_keys[scenario]].append(result.equilibrium_with_gravity_nm)

    # Compatibility aliases for older plotting/scripts that used pressure-oriented names.
    results["positive_gravity_pressure"] = results["positive_gravity"]
    results["negative_gravity_pressure"] = results["negative_gravity"]

    return results


__all__ = [
    "calculate_equilibrium_vs_thickness",
    "calculate_gravitational_pressure",
    "find_stable_equilibrium_distance_with_gravity",
    "find_zero_crossing_distance_with_gravity",
]
