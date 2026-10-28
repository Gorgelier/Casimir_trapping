"""Unified cached single-point trap analysis built only on evaluate_energy_curve."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np
from scipy.constants import Boltzmann

from .adapter import evaluate_energy_curve
from .barriers import TailAssessment, assess_tail, build_trap_candidates, select_trap_candidate
from .extrema import (
    boundary_minimum_at_start,
    detect_extremum_brackets,
    energy_resolution,
    merge_stationary_points,
    refine_extremum,
    secant_slopes,
    slope_zero_tolerance,
)
from .grids import build_distance_grid
from .models import (
    DistanceExtension,
    EnergyCurveResult,
    SinglePointResult,
    SweepPointInput,
    TrapAnalysisConfig,
)


EnergyEvaluator = Callable[[str, np.ndarray, dict[str, Any], str], EnergyCurveResult]


class EnergyEvaluationError(RuntimeError):
    """Contextual failure from an injected or real energy adapter."""


def reporting_conversion(geometry: str, temperature_k: float) -> tuple[float, str]:
    """Return native-to-kBT reporting factor without inventing finite extents."""
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature_K must be finite and positive for k_B T reporting units.")
    thermal = Boltzmann * temperature_k
    if geometry == "plate":
        return 1.0e-12 / thermal, "k_B T / um^2"
    if geometry == "cylinder":
        return 1.0e-6 / thermal, "k_B T / um"
    if geometry == "sphere":
        return 1.0 / thermal, "k_B T"
    raise ValueError(f"Unsupported geometry {geometry!r} for reporting conversion.")


def point_physical_parameters(point: SweepPointInput) -> dict[str, Any]:
    """Resolve adapter parameters from one immutable sweep point."""
    parameters = dict(point.physical_parameters)
    parameters.update(
        {
            "t_Au_m": point.t_Au_m,
            "t_PTFE_m": point.t_PTFE_m,
            "R_cyl_m": point.R_cyl_m,
            "R_sph_m": point.R_sph_m,
            "temperature_K": point.temperature_K,
        }
    )
    if "gravity_cases" not in parameters:
        if point.gravity_case in point.gravity_parameters:
            parameters["gravity_cases"] = point.gravity_parameters
        else:
            parameters["gravity_cases"] = {point.gravity_case: point.gravity_parameters}
    return parameters


class CachedEnergyEvaluator:
    """Exact-float cache for one point, batching only genuinely new separations."""

    def __init__(
        self,
        point: SweepPointInput,
        energy_evaluator: EnergyEvaluator,
        *,
        deadline: float | None = None,
    ) -> None:
        self.point = point
        self.energy_evaluator = energy_evaluator
        self.physical_parameters = point_physical_parameters(point)
        self.deadline = deadline
        self.adapter_call_count = 0
        self.actual_separation_evaluations = 0
        self.cache_hit_count = 0
        self.backend_runtime_s = 0.0
        self.backend_call_runtimes_s: list[float] = []
        self.energy_unit: str | None = None
        self.backend_metadata: dict[str, Any] = {}
        self._cache: dict[str, tuple[float, float, float, float]] = {}

    @property
    def evaluation_count(self) -> int:
        return self.actual_separation_evaluations

    @staticmethod
    def _key(distance_m: float) -> str:
        return float(distance_m).hex()

    def _backend_context(self, separation_points: int) -> str:
        return (
            f"geometry={self.point.geometry}, gravity={self.point.gravity_case}, "
            f"t_Au_m={self.point.t_Au_m}, t_PTFE_m={self.point.t_PTFE_m}, "
            f"R_cyl_m={self.point.R_cyl_m}, R_sph_m={self.point.R_sph_m}, "
            f"separation_points={separation_points}"
        )

    def _check_deadline(self) -> None:
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise TimeoutError("single-point analysis exceeded max_runtime_seconds")

    def evaluate_array(self, distances_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        distances = np.asarray(distances_m)
        if distances.ndim != 1 or distances.size == 0:
            raise EnergyEvaluationError("separation request must be a non-empty one-dimensional array")
        if np.iscomplexobj(distances):
            raise EnergyEvaluationError("separation request must be real-valued")
        distances = distances.astype(float, copy=False)
        if not np.all(np.isfinite(distances)) or np.any(distances <= 0.0):
            raise EnergyEvaluationError("separation request contains non-positive, NaN, or infinite values")
        keys = [self._key(value) for value in distances]
        missing_by_key: dict[str, float] = {}
        for key, value in zip(keys, distances):
            if key in self._cache or key in missing_by_key:
                self.cache_hit_count += 1
            else:
                missing_by_key[key] = float(value)
        if missing_by_key:
            self._check_deadline()
            missing = np.asarray(sorted(missing_by_key.values()), dtype=float)
            started = time.perf_counter()
            try:
                curve = self.energy_evaluator(
                    self.point.geometry,
                    missing,
                    self.physical_parameters,
                    self.point.gravity_case,
                )
            except Exception as exc:
                raise EnergyEvaluationError(
                    f"energy backend failed for {self._backend_context(missing.size)}: {exc}"
                ) from exc
            elapsed = time.perf_counter() - started
            self.adapter_call_count += 1
            self.backend_runtime_s += elapsed
            self.backend_call_runtimes_s.append(elapsed)
            self._check_deadline()
            required = ("distances_m", "interaction_energy", "gravity_energy", "total_energy", "energy_unit")
            if any(not hasattr(curve, name) for name in required):
                raise EnergyEvaluationError(
                    f"energy backend omitted a required component for {self._backend_context(missing.size)}"
                )
            returned_distances = np.asarray(curve.distances_m)
            components = [
                np.asarray(curve.interaction_energy),
                np.asarray(curve.gravity_energy),
                np.asarray(curve.total_energy),
            ]
            if returned_distances.shape != missing.shape or any(component.shape != missing.shape for component in components):
                shapes = [component.shape for component in components]
                raise EnergyEvaluationError(
                    f"energy backend shape mismatch for {self._backend_context(missing.size)}: "
                    f"distances={returned_distances.shape}, components={shapes}"
                )
            if np.iscomplexobj(returned_distances) or any(np.iscomplexobj(component) for component in components):
                raise EnergyEvaluationError(
                    f"energy backend returned non-real values for {self._backend_context(missing.size)}"
                )
            returned_distances = returned_distances.astype(float, copy=False)
            components = [component.astype(float, copy=False) for component in components]
            if not np.array_equal(returned_distances, missing):
                raise EnergyEvaluationError(
                    f"energy backend changed separation values/order for {self._backend_context(missing.size)}"
                )
            if any(not np.all(np.isfinite(component)) for component in components):
                raise EnergyEvaluationError(
                    f"energy backend returned NaN or infinity for {self._backend_context(missing.size)}"
                )
            self.energy_unit = str(curve.energy_unit)
            self.backend_metadata = dict(getattr(curve, "metadata", {}))
            for index, distance in enumerate(missing):
                self._cache[self._key(float(distance))] = (
                    float(distance),
                    float(components[0][index]),
                    float(components[1][index]),
                    float(components[2][index]),
                )
            self.actual_separation_evaluations += int(missing.size)
        interaction = np.asarray([self._cache[key][1] for key in keys], dtype=float)
        gravity = np.asarray([self._cache[key][2] for key in keys], dtype=float)
        total = np.asarray([self._cache[key][3] for key in keys], dtype=float)
        return interaction, gravity, total

    def evaluate_one(self, distance_m: float) -> tuple[float, float, float]:
        interaction, gravity, total = self.evaluate_array(np.asarray([distance_m], dtype=float))
        return float(interaction[0]), float(gravity[0]), float(total[0])

    def cached_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows = sorted(self._cache.values(), key=lambda row: row[0])
        if not rows:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty
        array = np.asarray(rows, dtype=float)
        return array[:, 0], array[:, 1], array[:, 2], array[:, 3]

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter_call_count": self.adapter_call_count,
            "actual_separation_evaluations": self.actual_separation_evaluations,
            "cache_hit_count": self.cache_hit_count,
            "backend_runtime_s": self.backend_runtime_s,
            "backend_call_runtimes_s": self.backend_call_runtimes_s,
            "backend": self.backend_metadata,
        }


def _initial_grid(point: SweepPointInput, config: TrapAnalysisConfig) -> np.ndarray:
    distance_spec = dict(point.distance_grid_settings)
    point_counts = dict(distance_spec.get("points", {}))
    point_counts[config.profile] = config.initial_grid_points
    distance_spec["points"] = point_counts
    return build_distance_grid(distance_spec, point.t_PTFE_m, config.profile)


def _extension_grid(current: np.ndarray, config: TrapAnalysisConfig) -> np.ndarray:
    old_max = float(current[-1])
    new_max = old_max * config.adaptive_extension.extension_factor
    log_density_points = int(
        np.ceil(
            (len(current) - 1)
            * np.log(config.adaptive_extension.extension_factor)
            / np.log(old_max / float(current[0]))
        )
    )
    new_count = max(config.adaptive_extension.tail_points, log_density_points)
    values = np.geomspace(old_max, new_max, new_count + 1, dtype=float)[1:]
    values[-1] = new_max
    return values


def _right_maximum_exists(minimum_index: int, brackets: list[Any]) -> bool:
    return any(bracket.kind == "maximum" and bracket.representative_index > minimum_index for bracket in brackets)


def _extension_reasons(
    distances: np.ndarray,
    total: np.ndarray,
    brackets: list[Any],
    tail: TailAssessment,
    config: TrapAnalysisConfig,
) -> list[str]:
    reasons: list[str] = []
    x = np.log(distances / distances[0])
    edge_start = (1.0 - config.adaptive_extension.edge_fraction) * x[-1]
    for bracket in brackets:
        if x[bracket.representative_index] >= edge_start:
            reasons.append(f"candidate_{bracket.kind}_in_far_edge")
        if bracket.right_index == len(distances) - 1:
            reasons.append(f"candidate_{bracket.kind}_bracket_touches_d_max")
    minima = [bracket for bracket in brackets if bracket.kind == "minimum"]
    _tail_x, sampled_slopes = secant_slopes(distances, total)
    if sampled_slopes[-1] < -slope_zero_tolerance(total, config.numerical_resolution):
        reasons.append("energy_descends_at_d_max")
    if any(not _right_maximum_exists(bracket.representative_index, brackets) for bracket in minima) and not tail.resolved:
        reasons.append("minimum_has_no_resolved_far_barrier")
    if not tail.resolved:
        reasons.append("far_tail_not_resolved")
    return sorted(set(reasons))


def _empty_failure_result(
    point: SweepPointInput,
    cache: CachedEnergyEvaluator,
    started: float,
    exc: Exception,
) -> SinglePointResult:
    distances, interaction, gravity, total = cache.cached_curve()
    factor, reporting_unit = reporting_conversion(point.geometry, point.temperature_K)
    return SinglePointResult(
        status="numerical_failure",
        geometry=point.geometry,
        gravity_case=point.gravity_case,
        input_parameters=point.to_dict(),
        distances_m=distances,
        interaction_energy=interaction,
        gravity_energy=gravity,
        total_energy=total,
        interaction_energy_reporting=interaction * factor,
        gravity_energy_reporting=gravity * factor,
        total_energy_reporting=total * factor,
        energy_unit=cache.energy_unit,
        reporting_unit=reporting_unit,
        trap_exists=None,
        publication_ready=False,
        warnings=[str(exc)],
        convergence_metadata=cache.metadata(),
        runtime_s=time.perf_counter() - started,
        exception_information={
            "type": type(exc).__name__,
            "message": str(exc),
            "geometry": point.geometry,
            "gravity_case": point.gravity_case,
            "evaluated_separation_points": cache.evaluation_count,
        },
    )


def _analyze_pass(
    point: SweepPointInput,
    analysis_config: TrapAnalysisConfig,
    *,
    energy_evaluator: EnergyEvaluator,
) -> SinglePointResult:
    started = time.perf_counter()
    deadline = None if analysis_config.max_runtime_seconds is None else started + analysis_config.max_runtime_seconds
    cache = CachedEnergyEvaluator(point, energy_evaluator, deadline=deadline)
    extensions: list[DistanceExtension] = []
    previous_plateau_estimate: float | None = None
    try:
        distances = _initial_grid(point, analysis_config)
        interaction, gravity, total = cache.evaluate_array(distances)
        extension_count = 0
        while True:
            brackets = detect_extremum_brackets(distances, total, analysis_config.numerical_resolution)
            tail = assess_tail(
                distances,
                interaction,
                gravity,
                total,
                tail_points=analysis_config.adaptive_extension.tail_points,
                resolution=analysis_config.numerical_resolution,
                previous_plateau_estimate=previous_plateau_estimate,
                extension_count=extension_count,
            )
            reasons = _extension_reasons(distances, total, brackets, tail, analysis_config)
            minima = [bracket for bracket in brackets if bracket.kind == "minimum"]
            all_minima_have_right_max = bool(minima) and all(
                _right_maximum_exists(bracket.representative_index, brackets) for bracket in minima
            )
            boundary_start = boundary_minimum_at_start(distances, total, analysis_config.numerical_resolution)
            if boundary_start and not minima:
                reasons = []
            if all_minima_have_right_max and not any("far_edge" in reason or "touches_d_max" in reason for reason in reasons):
                reasons = []
            if (
                tail.resolved
                and tail.kind != "open_far_escape"
                and not any("far_edge" in reason or "touches_d_max" in reason for reason in reasons)
            ):
                reasons = []
            if not reasons:
                break
            if (
                not analysis_config.adaptive_extension.enabled
                or extension_count >= analysis_config.adaptive_extension.max_extensions
            ):
                break
            if tail.plateau_energy_native is not None:
                previous_plateau_estimate = tail.plateau_energy_native
            new_points = _extension_grid(distances, analysis_config)
            previous_max = float(distances[-1])
            new_interaction, new_gravity, new_total = cache.evaluate_array(new_points)
            distances = np.concatenate((distances, new_points))
            interaction = np.concatenate((interaction, new_interaction))
            gravity = np.concatenate((gravity, new_gravity))
            total = np.concatenate((total, new_total))
            extension_count += 1
            extensions.append(
                DistanceExtension(
                    extension_index=extension_count,
                    previous_d_max_m=previous_max,
                    new_d_max_m=float(distances[-1]),
                    reasons=reasons,
                    new_point_count=len(new_points),
                )
            )

        brackets = detect_extremum_brackets(distances, total, analysis_config.numerical_resolution)
        tail = assess_tail(
            distances,
            interaction,
            gravity,
            total,
            tail_points=analysis_config.adaptive_extension.tail_points,
            resolution=analysis_config.numerical_resolution,
            previous_plateau_estimate=previous_plateau_estimate,
            extension_count=len(extensions),
        )
        stationary = [
            refine_extremum(
                bracket,
                distances,
                interaction,
                gravity,
                total,
                cache,
                analysis_config.extrema_refinement,
                analysis_config.numerical_resolution,
            )
            for bracket in brackets
        ]
        stationary = merge_stationary_points(
            stationary,
            analysis_config.extrema_refinement.merge_distance_rtol,
        )
        # Refinement samples remain cached for diagnostics but the primary curve
        # retains the sorted initial/extension grid used for detection.
        factor, reporting_unit = reporting_conversion(point.geometry, point.temperature_K)
        candidates = build_trap_candidates(
            stationary,
            distances,
            total,
            tail,
            analysis_config.numerical_resolution,
            factor,
        )
        selected_index, global_index, largest_index, selection_warnings = select_trap_candidate(
            candidates,
            analysis_config.numerical_resolution,
        )
        minima = [point_value for point_value in stationary if point_value.kind == "minimum"]
        stable = [candidate for candidate in candidates if candidate.stable]
        boundary_start = boundary_minimum_at_start(distances, total, analysis_config.numerical_resolution)
        tolerance = energy_resolution(total, analysis_config.numerical_resolution)
        boundary_is_global = boundary_start and (
            not minima or float(total[0]) < min(point_value.total_energy_native for point_value in minima) - tolerance
        )
        unresolved_candidates = any(
            candidate.near_barrier.unresolved
            or candidate.far_barrier.unresolved
            or not candidate.minimum.refinement_success
            for candidate in candidates
        )
        # A lower endpoint is a competing boundary state, but it does not erase
        # an interior metastable well protected in both escape directions.
        # ``boundary_minimum`` is reserved for curves with no stable interior
        # candidate (most commonly a monotonic increase from d_min).
        if boundary_is_global and not stable:
            status = "boundary_minimum"
            trap_exists: bool | None = False
            selected_index = None
        elif len(stable) >= 2:
            status = "multiple_traps"
            trap_exists = True
        elif len(stable) == 1:
            status = "success"
            trap_exists = True
        elif not tail.resolved and (minima or not boundary_start):
            status = "unresolved_tail"
            trap_exists = None
            selected_index = None
        elif unresolved_candidates:
            status = "numerically_unresolved"
            trap_exists = None
            selected_index = None
        else:
            status = "no_trap"
            trap_exists = False
            selected_index = None

        selected = None if selected_index is None else candidates[selected_index]
        cutoff_sensitive = bool(selected.cutoff_sensitive) if selected is not None else False
        warnings = [*selection_warnings]
        if cutoff_sensitive:
            near_cutoff_nm = float(distances[0]) * 1.0e9
            warnings.append(
                f"selected trap depends on the {near_cutoff_nm:g} nm near cutoff; "
                "the current campaign does not "
                "enforce a model-validity lower bound, so physical validity there is not established"
            )
        if status == "unresolved_tail":
            warnings.extend(tail.warnings)
        return SinglePointResult(
            status=status,
            geometry=point.geometry,
            gravity_case=point.gravity_case,
            input_parameters=point.to_dict(),
            distances_m=distances,
            interaction_energy=interaction,
            gravity_energy=gravity,
            total_energy=total,
            interaction_energy_reporting=interaction * factor,
            gravity_energy_reporting=gravity * factor,
            total_energy_reporting=total * factor,
            energy_unit=cache.energy_unit,
            reporting_unit=reporting_unit,
            trap_exists=trap_exists,
            stationary_points=stationary,
            local_minimum_count=len(minima),
            stable_trap_count=len(stable),
            trap_candidates=candidates,
            selected_trap_index=selected_index,
            selected_trap_rule="global_minimum_then_largest_trap_depth_then_smaller_d" if selected is not None else None,
            global_energy_minimum_index=global_index,
            largest_barrier_index=largest_index,
            d_trap_m=None if selected is None else selected.minimum.d_m,
            delta_u_native=None if selected is None else selected.delta_u_native,
            delta_u_reporting=None if selected is None else selected.delta_u_reporting,
            delta_u_infinite=False if selected is None else selected.delta_u_infinite,
            cutoff_sensitive=cutoff_sensitive,
            publication_ready=bool(selected is not None and selected.publication_ready),
            distance_extensions=extensions,
            trap_classification="not_evaluated" if selected is None else selected.stability_class,
            trap_count=len(stable),
            delta_U=None if selected is None else selected.delta_u_native,
            near_barrier=None if selected is None else selected.near_barrier.height_native,
            far_barrier=None if selected is None else selected.far_barrier.height_native,
            warnings=warnings,
            convergence_metadata={
                **cache.metadata(),
                "initial_d_min_m": float(_initial_grid(point, analysis_config)[0]),
                "initial_d_max_m": float(_initial_grid(point, analysis_config)[-1]),
                "final_d_max_m": float(distances[-1]),
                "finite_energy_fraction": float(
                    np.count_nonzero(
                        np.isfinite(interaction) & np.isfinite(gravity) & np.isfinite(total)
                    )
                    / len(distances)
                ),
                "tail_kind": tail.kind,
                "tail_plateau_energy_native": tail.plateau_energy_native,
                "tail_plateau_uncertainty_native": tail.plateau_uncertainty_native,
            },
            runtime_s=time.perf_counter() - started,
        )
    except Exception as exc:
        return _empty_failure_result(point, cache, started, exc)


def analyze_single_point(
    point: SweepPointInput,
    analysis_config: TrapAnalysisConfig,
    *,
    energy_evaluator: EnergyEvaluator = evaluate_energy_curve,
    check_convergence: bool = False,
) -> SinglePointResult:
    """Analyze one parameter point; never dispatch a parameter series."""
    require_convergence = bool(check_convergence or analysis_config.convergence.enabled)
    base_config = replace(
        analysis_config,
        convergence=replace(analysis_config.convergence, enabled=False),
    )
    base = _analyze_pass(point, base_config, energy_evaluator=energy_evaluator)
    if not require_convergence or base.status == "numerical_failure":
        return base
    from .convergence import run_convergence_check

    return run_convergence_check(
        point,
        analysis_config,
        base,
        energy_evaluator=energy_evaluator,
    )


__all__ = [
    "CachedEnergyEvaluator",
    "EnergyEvaluationError",
    "analyze_single_point",
    "point_physical_parameters",
    "reporting_conversion",
]
