"""Scale-aware stationary-point detection and cached backend refinement in log d."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from scipy import optimize

from .models import ExtremaRefinementConfig, NumericalResolutionConfig, StationaryPoint


@dataclass(frozen=True)
class ExtremumBracket:
    kind: Literal["minimum", "maximum"]
    left_index: int
    representative_index: int
    right_index: int
    plateau: bool


class PointEnergyEvaluator(Protocol):
    @property
    def evaluation_count(self) -> int: ...

    def evaluate_one(self, distance_m: float) -> tuple[float, float, float]: ...


def energy_resolution(values: np.ndarray, config: NumericalResolutionConfig) -> float:
    """Return a native-unit comparison scale for *energy differences*.

    Extrema and barrier heights are invariant under ``U -> U + constant``.  The
    configurable relative tolerance must therefore scale with the span of the
    supplied values, not their arbitrary absolute reference level.  The
    floating-point term still accounts for cancellation when a large common
    offset has already limited the precision of the represented values.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return max(config.energy_atol_native, np.finfo(float).tiny)
    magnitude = float(np.max(np.abs(array)))
    span = float(np.ptp(array))
    floating = np.finfo(float).eps * max(magnitude, span, np.finfo(float).tiny) * 64.0
    relative = config.energy_comparison_rtol * max(span, np.finfo(float).tiny)
    return max(config.energy_atol_native, relative, floating)


def slope_zero_tolerance(values: np.ndarray, config: NumericalResolutionConfig) -> float:
    array = np.asarray(values, dtype=float)
    # A constant shift of the potential cannot change whether its slope is
    # zero.  Use only the variation of the curve for the relative part.
    scale = max(float(np.ptp(array)), np.finfo(float).tiny)
    return max(config.slope_zero_rtol * scale, energy_resolution(array, config))


def secant_slopes(distances_m: np.ndarray, total_energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(distances_m, dtype=float)
    values = np.asarray(total_energy, dtype=float)
    if distances.ndim != 1 or values.ndim != 1 or distances.size != values.size or distances.size < 2:
        raise ValueError("distances_m and total_energy must be same-length one-dimensional arrays with at least two points.")
    if np.any(distances <= 0.0) or np.any(np.diff(distances) <= 0.0):
        raise ValueError("distances_m must be positive and strictly increasing.")
    x = np.log(distances / distances[0])
    slopes = np.diff(values) / np.diff(x)
    return x, slopes


def detect_extremum_brackets(
    distances_m: np.ndarray,
    total_energy: np.ndarray,
    config: NumericalResolutionConfig,
) -> list[ExtremumBracket]:
    """Detect sign changes, collapsing any near-zero-slope plateau to one bracket."""
    _x, slopes = secant_slopes(distances_m, total_energy)
    tolerance = slope_zero_tolerance(total_energy, config)
    signs = np.zeros(slopes.size, dtype=np.int8)
    signs[slopes > tolerance] = 1
    signs[slopes < -tolerance] = -1
    significant = np.flatnonzero(signs)
    brackets: list[ExtremumBracket] = []
    for left_slope_index, right_slope_index in zip(significant[:-1], significant[1:]):
        left_sign = int(signs[left_slope_index])
        right_sign = int(signs[right_slope_index])
        if left_sign == right_sign:
            continue
        if left_sign < 0 < right_sign:
            kind: Literal["minimum", "maximum"] = "minimum"
        elif left_sign > 0 > right_sign:
            kind = "maximum"
        else:
            continue
        left_index = int(left_slope_index)
        right_index = int(right_slope_index + 1)
        representative = int(round((left_slope_index + 1 + right_slope_index) / 2.0))
        brackets.append(
            ExtremumBracket(
                kind=kind,
                left_index=left_index,
                representative_index=representative,
                right_index=right_index,
                plateau=bool(right_slope_index > left_slope_index + 1),
            )
        )
    return brackets


def boundary_minimum_at_start(
    distances_m: np.ndarray,
    total_energy: np.ndarray,
    config: NumericalResolutionConfig,
) -> bool:
    _x, slopes = secant_slopes(distances_m, total_energy)
    tolerance = slope_zero_tolerance(total_energy, config)
    significant = slopes[np.abs(slopes) > tolerance]
    return bool(significant.size and significant[0] > 0.0)


def descending_at_end(
    distances_m: np.ndarray,
    total_energy: np.ndarray,
    config: NumericalResolutionConfig,
) -> bool:
    _x, slopes = secant_slopes(distances_m, total_energy)
    tolerance = slope_zero_tolerance(total_energy, config)
    significant = slopes[np.abs(slopes) > tolerance]
    return bool(significant.size and significant[-1] < 0.0)


def _sampled_stationary_point(
    bracket: ExtremumBracket,
    distances_m: np.ndarray,
    interaction: np.ndarray,
    gravity: np.ndarray,
    total: np.ndarray,
    *,
    warning: str,
) -> StationaryPoint:
    index = bracket.representative_index
    x = np.log(distances_m / distances_m[0])
    left_slope = (total[index] - total[bracket.left_index]) / (x[index] - x[bracket.left_index])
    right_slope = (total[bracket.right_index] - total[index]) / (x[bracket.right_index] - x[index])
    width_left = x[index] - x[bracket.left_index]
    width_right = x[bracket.right_index] - x[index]
    curvature = 2.0 * (right_slope - left_slope) / (width_left + width_right)
    return StationaryPoint(
        kind=bracket.kind,
        d_m=float(distances_m[index]),
        x=float(x[index]),
        interaction_energy_native=float(interaction[index]),
        gravity_energy_native=float(gravity[index]),
        total_energy_native=float(total[index]),
        bracket_d_m=(float(distances_m[bracket.left_index]), float(distances_m[bracket.right_index])),
        left_slope=float(left_slope),
        right_slope=float(right_slope),
        curvature_x=float(curvature),
        refinement_method="sampled_bracket",
        refinement_success=False,
        evaluation_count=0,
        warnings=[warning],
    )


def refine_extremum(
    bracket: ExtremumBracket,
    distances_m: np.ndarray,
    interaction: np.ndarray,
    gravity: np.ndarray,
    total: np.ndarray,
    evaluator: PointEnergyEvaluator,
    refinement: ExtremaRefinementConfig,
    resolution: NumericalResolutionConfig,
) -> StationaryPoint:
    """Refine one bracket with bounded real-backend evaluations and verify it."""
    if not refinement.enabled:
        return _sampled_stationary_point(
            bracket,
            distances_m,
            interaction,
            gravity,
            total,
            warning="extrema refinement is disabled",
        )
    d_reference = float(distances_m[0])
    x_values = np.log(distances_m / d_reference)
    x_left = float(x_values[bracket.left_index])
    x_right = float(x_values[bracket.right_index])
    before = evaluator.evaluation_count

    def objective(x_value: float) -> float:
        _interaction, _gravity, total_value = evaluator.evaluate_one(d_reference * float(np.exp(x_value)))
        return total_value if bracket.kind == "minimum" else -total_value

    result = optimize.minimize_scalar(
        objective,
        bounds=(x_left, x_right),
        method="bounded",
        options={"xatol": refinement.xatol, "maxiter": refinement.max_evaluations},
    )
    warnings: list[str] = []
    x_star = float(result.x)
    bracket_width = x_right - x_left
    interior_margin = max(refinement.xatol, np.finfo(float).eps * max(abs(x_star), 1.0) * 32.0)
    solver_ok = bool(result.success) and x_left + interior_margin < x_star < x_right - interior_margin
    if not result.success:
        warnings.append(f"bounded refinement failed: {result.message}")
    if not (x_left + interior_margin < x_star < x_right - interior_margin):
        warnings.append("refined stationary point is not strictly inside its bracket")

    validation_step = min(
        0.2 * bracket_width,
        max(2.0 * refinement.xatol, bracket_width * 1.0e-3),
    )
    if validation_step <= 0.0:
        return _sampled_stationary_point(
            bracket,
            distances_m,
            interaction,
            gravity,
            total,
            warning="refinement bracket has no positive log-coordinate width",
        )
    x_minus = max(x_left, x_star - validation_step)
    x_plus = min(x_right, x_star + validation_step)
    d_star = d_reference * float(np.exp(x_star))
    d_minus = d_reference * float(np.exp(x_minus))
    d_plus = d_reference * float(np.exp(x_plus))
    i_star, g_star, u_star = evaluator.evaluate_one(d_star)
    _i_minus, _g_minus, u_minus = evaluator.evaluate_one(d_minus)
    _i_plus, _g_plus, u_plus = evaluator.evaluate_one(d_plus)
    left_slope = (u_star - u_minus) / (x_star - x_minus)
    right_slope = (u_plus - u_star) / (x_plus - x_star)
    curvature = 2.0 * (right_slope - left_slope) / (x_plus - x_minus)
    tolerance = energy_resolution(np.asarray([u_minus, u_star, u_plus]), resolution)
    # Refinement verifies a stationary point from its immediate neighbourhood.
    # Scaling this tolerance with the full curve can let a large near-contact
    # energy range mask a genuine shallow far-field extremum.
    slope_tolerance = slope_zero_tolerance(np.asarray([u_minus, u_star, u_plus]), resolution)

    if bracket.kind == "minimum":
        order_ok = u_star <= u_minus + tolerance and u_star <= u_plus + tolerance
        signs_ok = left_slope < -slope_tolerance and right_slope > slope_tolerance
        curvature_ok = curvature > 0.0
    else:
        order_ok = u_star >= u_minus - tolerance and u_star >= u_plus - tolerance
        signs_ok = left_slope > slope_tolerance and right_slope < -slope_tolerance
        curvature_ok = curvature < 0.0
    if not order_ok:
        warnings.append("refined energy ordering does not verify the stationary-point kind")
    if not signs_ok:
        warnings.append("refined left/right secant signs do not verify the stationary-point kind")
    if not curvature_ok:
        warnings.append("refined local curvature has the wrong sign")
    success = solver_ok and order_ok and signs_ok and curvature_ok
    return StationaryPoint(
        kind=bracket.kind,
        d_m=float(d_star),
        x=float(x_star),
        interaction_energy_native=float(i_star),
        gravity_energy_native=float(g_star),
        total_energy_native=float(u_star),
        bracket_d_m=(float(distances_m[bracket.left_index]), float(distances_m[bracket.right_index])),
        left_slope=float(left_slope),
        right_slope=float(right_slope),
        curvature_x=float(curvature),
        refinement_method=refinement.method,
        refinement_success=success,
        evaluation_count=evaluator.evaluation_count - before,
        warnings=warnings,
    )


def merge_stationary_points(points: list[StationaryPoint], distance_rtol: float) -> list[StationaryPoint]:
    """Merge only same-kind refined points that are extremely close in d."""
    merged: list[StationaryPoint] = []
    for point in sorted(points, key=lambda item: item.d_m):
        if not merged or point.kind != merged[-1].kind or not np.isclose(
            point.d_m,
            merged[-1].d_m,
            rtol=distance_rtol,
            atol=0.0,
        ):
            merged.append(point)
            continue
        incumbent = merged[-1]
        incumbent_score = (incumbent.refinement_success, -abs(incumbent.left_slope + incumbent.right_slope))
        point_score = (point.refinement_success, -abs(point.left_slope + point.right_slope))
        if point_score > incumbent_score:
            merged[-1] = point
    return merged


__all__ = [
    "ExtremumBracket",
    "boundary_minimum_at_start",
    "descending_at_end",
    "detect_extremum_brackets",
    "energy_resolution",
    "merge_stationary_points",
    "refine_extremum",
    "secant_slopes",
    "slope_zero_tolerance",
]
