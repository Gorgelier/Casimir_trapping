"""Far-tail assessment, escape-barrier classification, and trap selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extrema import energy_resolution, secant_slopes, slope_zero_tolerance
from .models import (
    EscapeBarrier,
    NumericalResolutionConfig,
    StationaryPoint,
    TrapCandidate,
)


@dataclass(frozen=True)
class TailAssessment:
    kind: str
    resolved: bool
    protected: bool
    plateau_energy_native: float | None
    plateau_uncertainty_native: float | None
    total_slope_native: float
    gravity_slope_native: float
    interaction_slope_native: float
    warnings: list[str]


def assess_tail(
    distances_m: np.ndarray,
    interaction: np.ndarray,
    gravity: np.ndarray,
    total: np.ndarray,
    *,
    tail_points: int,
    resolution: NumericalResolutionConfig,
    previous_plateau_estimate: float | None,
    extension_count: int,
) -> TailAssessment:
    """Classify a sampled far tail without extrapolating through unresolved data."""
    count = min(tail_points, len(distances_m))
    if count < 3:
        return TailAssessment("unresolved_boundary", False, False, None, None, 0.0, 0.0, 0.0, ["fewer than three tail points"])
    tail_d = distances_m[-count:]
    tail_i = interaction[-count:]
    tail_g = gravity[-count:]
    tail_u = total[-count:]
    _x, total_slopes_x = secant_slopes(tail_d, tail_u)
    total_slopes = np.diff(tail_u) / np.diff(tail_d)
    gravity_slopes = np.diff(tail_g) / np.diff(tail_d)
    interaction_slopes = np.diff(tail_i) / np.diff(tail_d)
    tolerance_x = slope_zero_tolerance(total, resolution)
    distance_scale = max(float(np.median(tail_d)), np.finfo(float).tiny)
    tolerance = tolerance_x / distance_scale
    total_median = float(np.median(total_slopes))
    gravity_median = float(np.median(gravity_slopes))
    interaction_median = float(np.median(interaction_slopes))
    gravity_zero = bool(np.all(np.abs(gravity_slopes) <= tolerance))
    # Tail classification, like every barrier comparison, must be invariant to
    # the arbitrary zero of potential energy.  Scale relative tolerances with
    # energy variation rather than max(abs(U)).
    global_scale = max(float(np.ptp(total)), resolution.energy_atol_native, np.finfo(float).tiny)
    flat = float(np.ptp(tail_u)) <= max(
        resolution.tail_flatness_rtol * global_scale,
        energy_resolution(tail_u, resolution),
    )
    plateau = float(np.median(tail_u))
    uncertainty = 0.5 * float(np.ptp(tail_u))
    extension_stable = (
        previous_plateau_estimate is not None
        and abs(plateau - previous_plateau_estimate)
        <= max(
            resolution.tail_extension_rtol * global_scale,
            energy_resolution(np.asarray([plateau, previous_plateau_estimate]), resolution),
        )
    )
    if gravity_zero and flat and extension_count >= 1 and extension_stable:
        return TailAssessment(
            "asymptotic_plateau",
            True,
            True,
            plateau,
            uncertainty,
            total_median,
            gravity_median,
            interaction_median,
            [],
        )

    gravity_scale = max(abs(gravity_median), tolerance)
    gravity_constant = float(np.ptp(gravity_slopes)) <= resolution.tail_flatness_rtol * gravity_scale
    interaction_decreasing = abs(float(interaction_slopes[-1])) <= abs(float(interaction_slopes[0])) + tolerance
    if (
        gravity_median > tolerance
        and gravity_constant
        and np.all(total_slopes > tolerance)
        and abs(interaction_median) < gravity_median / resolution.gravity_dominance_ratio
        and interaction_decreasing
    ):
        return TailAssessment(
            "monotonic_confinement",
            True,
            True,
            None,
            None,
            total_median,
            gravity_median,
            interaction_median,
            [],
        )

    if np.all(total_slopes < -tolerance) and np.all(total_slopes_x < -tolerance_x):
        return TailAssessment(
            "open_far_escape",
            True,
            False,
            None,
            None,
            total_median,
            gravity_median,
            interaction_median,
            [],
        )
    return TailAssessment(
        "unresolved_boundary",
        False,
        False,
        plateau if gravity_zero else None,
        uncertainty if gravity_zero else None,
        total_median,
        gravity_median,
        interaction_median,
        ["far-tail slopes or plateau estimate have not stabilized"],
    )


def _reporting(value: float | None, factor: float) -> float | None:
    return None if value is None else value * factor


def near_escape_barrier(
    minimum: StationaryPoint,
    maxima: list[StationaryPoint],
    distances_m: np.ndarray,
    total: np.ndarray,
    resolution: NumericalResolutionConfig,
    reporting_factor: float,
) -> EscapeBarrier:
    left_maxima = [point for point in maxima if point.d_m < minimum.d_m]
    if left_maxima:
        barrier_point = max(left_maxima, key=lambda point: point.d_m)
        if not barrier_point.refinement_success:
            return EscapeBarrier(
                side="near",
                kind="unresolved_boundary",
                location_d_m=barrier_point.d_m,
                energy_native=barrier_point.total_energy_native,
                height_native=None,
                height_reporting=None,
                protected=False,
                finite=True,
                unresolved=True,
                warnings=["nearest near-side maximum failed numerical refinement"],
            )
        height = barrier_point.total_energy_native - minimum.total_energy_native
        tolerance = energy_resolution(
            np.asarray([barrier_point.total_energy_native, minimum.total_energy_native]),
            resolution,
        )
        protected = bool(height > tolerance)
        return EscapeBarrier(
            side="near",
            kind="interior_maximum" if protected else "open_contact_escape",
            location_d_m=barrier_point.d_m,
            energy_native=barrier_point.total_energy_native,
            height_native=height if protected else None,
            height_reporting=_reporting(height, reporting_factor) if protected else None,
            protected=protected,
            finite=True,
            warnings=[] if protected else ["near-side maximum is not above numerical energy resolution"],
        )

    left_mask = distances_m < minimum.d_m
    left_d = distances_m[left_mask]
    left_u = total[left_mask]
    if left_d.size < 2:
        return EscapeBarrier(
            side="near",
            kind="unresolved_boundary",
            location_d_m=None,
            energy_native=None,
            height_native=None,
            height_reporting=None,
            protected=False,
            finite=True,
            unresolved=True,
            warnings=["too few samples between d_min and the minimum"],
        )
    tolerance = energy_resolution(np.concatenate((left_u, [minimum.total_energy_native])), resolution)
    maximum_index = int(np.argmax(left_u))
    maximum_energy = float(left_u[maximum_index])
    height = maximum_energy - minimum.total_energy_native
    combined = np.concatenate((left_u, [minimum.total_energy_native]))
    slopes = np.diff(combined)
    mostly_descending_toward_minimum = bool(np.all(slopes <= tolerance))
    if height > tolerance and mostly_descending_toward_minimum:
        return EscapeBarrier(
            side="near",
            kind="cutoff_boundary",
            location_d_m=float(left_d[maximum_index]),
            energy_native=maximum_energy,
            height_native=height,
            height_reporting=_reporting(height, reporting_factor),
            protected=True,
            finite=True,
            lower_bound=True,
            warnings=["near barrier is a lower bound at the configured d_min cutoff"],
        )
    if float(left_u[0]) < minimum.total_energy_native - tolerance or np.all(slopes >= -tolerance):
        return EscapeBarrier(
            side="near",
            kind="open_contact_escape",
            location_d_m=float(left_d[0]),
            energy_native=float(left_u[0]),
            height_native=None,
            height_reporting=None,
            protected=False,
            finite=True,
            warnings=["energy admits a decreasing escape path toward contact"],
        )
    return EscapeBarrier(
        side="near",
        kind="unresolved_boundary",
        location_d_m=float(left_d[0]),
        energy_native=float(left_u[0]),
        height_native=None,
        height_reporting=None,
        protected=False,
        finite=True,
        unresolved=True,
        warnings=["near-side trend is below numerical resolution or non-monotonic"],
    )


def far_escape_barrier(
    minimum: StationaryPoint,
    maxima: list[StationaryPoint],
    tail: TailAssessment,
    resolution: NumericalResolutionConfig,
    reporting_factor: float,
) -> EscapeBarrier:
    right_maxima = [point for point in maxima if point.d_m > minimum.d_m]
    if right_maxima:
        barrier_point = min(right_maxima, key=lambda point: point.d_m)
        if not barrier_point.refinement_success:
            return EscapeBarrier(
                side="far",
                kind="unresolved_boundary",
                location_d_m=barrier_point.d_m,
                energy_native=barrier_point.total_energy_native,
                height_native=None,
                height_reporting=None,
                protected=False,
                finite=True,
                unresolved=True,
                warnings=["nearest far-side maximum failed numerical refinement"],
            )
        height = barrier_point.total_energy_native - minimum.total_energy_native
        tolerance = energy_resolution(
            np.asarray([barrier_point.total_energy_native, minimum.total_energy_native]),
            resolution,
        )
        protected = bool(height > tolerance)
        return EscapeBarrier(
            side="far",
            kind="interior_maximum" if protected else "open_far_escape",
            location_d_m=barrier_point.d_m,
            energy_native=barrier_point.total_energy_native,
            height_native=height if protected else None,
            height_reporting=_reporting(height, reporting_factor) if protected else None,
            protected=protected,
            finite=True,
            warnings=[] if protected else ["far-side maximum is not above numerical energy resolution"],
        )
    if tail.kind == "asymptotic_plateau" and tail.plateau_energy_native is not None:
        height = tail.plateau_energy_native - minimum.total_energy_native
        tolerance = max(
            energy_resolution(np.asarray([tail.plateau_energy_native, minimum.total_energy_native]), resolution),
            tail.plateau_uncertainty_native or 0.0,
        )
        if height > tolerance:
            return EscapeBarrier(
                side="far",
                kind="asymptotic_plateau",
                location_d_m=None,
                energy_native=tail.plateau_energy_native,
                height_native=height,
                height_reporting=_reporting(height, reporting_factor),
                protected=True,
                finite=True,
                uncertainty_native=tail.plateau_uncertainty_native,
            )
        return EscapeBarrier(
            side="far",
            kind="open_far_escape",
            location_d_m=None,
            energy_native=tail.plateau_energy_native,
            height_native=None,
            height_reporting=None,
            protected=False,
            finite=True,
            uncertainty_native=tail.plateau_uncertainty_native,
            warnings=["far plateau is not above the minimum by the numerical resolution"],
        )
    if tail.kind == "monotonic_confinement":
        return EscapeBarrier(
            side="far",
            kind="monotonic_confinement",
            location_d_m=None,
            energy_native=None,
            height_native=None,
            height_reporting=None,
            protected=True,
            finite=False,
            is_infinite=True,
        )
    if tail.kind == "open_far_escape":
        return EscapeBarrier(
            side="far",
            kind="open_far_escape",
            location_d_m=None,
            energy_native=None,
            height_native=None,
            height_reporting=None,
            protected=False,
            finite=True,
        )
    return EscapeBarrier(
        side="far",
        kind="unresolved_boundary",
        location_d_m=None,
        energy_native=tail.plateau_energy_native,
        height_native=None,
        height_reporting=None,
        protected=False,
        finite=True,
        unresolved=True,
        uncertainty_native=tail.plateau_uncertainty_native,
        warnings=list(tail.warnings),
    )


def build_trap_candidates(
    stationary_points: list[StationaryPoint],
    distances_m: np.ndarray,
    total_energy: np.ndarray,
    tail: TailAssessment,
    resolution: NumericalResolutionConfig,
    reporting_factor: float,
) -> list[TrapCandidate]:
    minima = sorted((point for point in stationary_points if point.kind == "minimum"), key=lambda point: point.d_m)
    maxima = sorted((point for point in stationary_points if point.kind == "maximum"), key=lambda point: point.d_m)
    candidates: list[TrapCandidate] = []
    for index, minimum in enumerate(minima):
        near = near_escape_barrier(minimum, maxima, distances_m, total_energy, resolution, reporting_factor)
        far = far_escape_barrier(minimum, maxima, tail, resolution, reporting_factor)
        warnings = [*minimum.warnings, *near.warnings, *far.warnings]
        cutoff_sensitive = near.kind == "cutoff_boundary"
        stable = minimum.refinement_success and near.protected and far.protected and not near.unresolved and not far.unresolved
        delta_native: float | None = None
        delta_reporting: float | None = None
        delta_infinite = False
        lower_bound = False
        if not minimum.refinement_success:
            stability = "numerically_unresolved"
        elif near.unresolved or far.unresolved:
            stability = "unresolved_tail" if far.unresolved else "numerically_unresolved"
            stable = False
        elif not near.protected:
            stability = "no_trap_open_contact" if near.kind == "open_contact_escape" else "no_trap_unresolved_shallow_minimum"
            stable = False
        elif not far.protected:
            stability = "no_trap_open_far" if far.kind == "open_far_escape" else "no_trap_unresolved_shallow_minimum"
            stable = False
        else:
            finite_heights = [
                barrier.height_native
                for barrier in (near, far)
                if barrier.finite and barrier.height_native is not None
            ]
            if not finite_heights:
                delta_infinite = True
            else:
                delta_native = float(min(finite_heights))
                delta_reporting = delta_native * reporting_factor
                tolerance = energy_resolution(
                    np.asarray([minimum.total_energy_native, minimum.total_energy_native + delta_native]),
                    resolution,
                )
                if delta_native <= tolerance:
                    stable = False
                    delta_native = None
                    delta_reporting = None
                    stability = "no_trap_unresolved_shallow_minimum"
            if stable:
                lower_bound = bool((near.lower_bound and near.height_native == delta_native) or (far.lower_bound and far.height_native == delta_native))
                if cutoff_sensitive:
                    stability = "stable_cutoff_limited"
                    warnings.append("1 nm model-validity lower bound is unresolved; cutoff-limited trap is not publication-ready")
                elif far.kind == "monotonic_confinement":
                    stability = "stable_monotonic_far_confinement"
                else:
                    stability = "stable_finite_barriers"
        candidates.append(
            TrapCandidate(
                candidate_index=index,
                minimum=minimum,
                near_barrier=near,
                far_barrier=far,
                stable=stable,
                stability_class=stability,
                delta_u_native=delta_native if stable else None,
                delta_u_reporting=delta_reporting if stable else None,
                delta_u_is_lower_bound=lower_bound if stable else False,
                delta_u_infinite=delta_infinite if stable else False,
                cutoff_sensitive=cutoff_sensitive,
                publication_ready=bool(stable and not cutoff_sensitive),
                branch_order=index,
                warnings=warnings,
            )
        )
    return candidates


def select_trap_candidate(
    candidates: list[TrapCandidate],
    resolution: NumericalResolutionConfig,
) -> tuple[int | None, int | None, int | None, list[str]]:
    """Select global-energy minimum, then depth, then smaller d among stable traps."""
    stable = [candidate for candidate in candidates if candidate.stable]
    if not stable:
        return None, None, None, []
    global_candidate = min(stable, key=lambda candidate: candidate.minimum.total_energy_native)
    global_index = global_candidate.candidate_index

    def depth_key(candidate: TrapCandidate) -> float:
        if candidate.delta_u_infinite:
            return float("inf")
        return -float("inf") if candidate.delta_u_native is None else candidate.delta_u_native

    largest = max(stable, key=depth_key)
    largest_index = largest.candidate_index
    energy_values = np.asarray([candidate.minimum.total_energy_native for candidate in stable])
    tolerance = energy_resolution(energy_values, resolution)
    # When two refined minima are almost degenerate, the span of just those
    # minima can be much smaller than the numerical scale of either well.  Use
    # the (translation-invariant) resolved barrier depths as an additional
    # comparison scale instead of the absolute potential-energy reference.
    finite_depths = [
        abs(float(candidate.delta_u_native))
        for candidate in stable
        if candidate.delta_u_native is not None and np.isfinite(candidate.delta_u_native)
    ]
    if finite_depths:
        tolerance = max(
            tolerance,
            resolution.energy_comparison_rtol * max(finite_depths),
        )
    tied = [
        candidate
        for candidate in stable
        if abs(candidate.minimum.total_energy_native - global_candidate.minimum.total_energy_native) <= tolerance
    ]
    warnings: list[str] = []
    if len(tied) > 1:
        best_depth = max(depth_key(candidate) for candidate in tied)
        depth_tied = [
            candidate
            for candidate in tied
            if (np.isinf(best_depth) and candidate.delta_u_infinite)
            or (
                not np.isinf(best_depth)
                and candidate.delta_u_native is not None
                and abs(candidate.delta_u_native - best_depth) <= tolerance
            )
        ]
        selected = min(depth_tied, key=lambda candidate: candidate.minimum.d_m)
        if len(depth_tied) > 1:
            warnings.append("selection tie remained after energy and trap-depth comparison; smaller d was selected")
    else:
        selected = global_candidate
    return selected.candidate_index, global_index, largest_index, warnings


__all__ = [
    "TailAssessment",
    "assess_tail",
    "build_trap_candidates",
    "far_escape_barrier",
    "near_escape_barrier",
    "select_trap_candidate",
]
