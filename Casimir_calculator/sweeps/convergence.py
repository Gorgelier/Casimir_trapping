"""Independent base/refined-grid convergence checks for one parameter point."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from .models import ConvergenceResult, EnergyCurveResult, SinglePointResult, SweepPointInput, TrapAnalysisConfig


EnergyEvaluator = Callable[[str, np.ndarray, dict[str, Any], str], EnergyCurveResult]


def _safe_relative_change(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    scale = max(abs(first), abs(second), np.finfo(float).tiny)
    return abs(second - first) / scale


def _selected_barrier_signature(result: SinglePointResult) -> tuple[str, str, bool, bool] | None:
    if result.selected_trap_index is None:
        return None
    candidate = result.trap_candidates[result.selected_trap_index]
    return (
        candidate.near_barrier.kind,
        candidate.far_barrier.kind,
        candidate.near_barrier.protected,
        candidate.far_barrier.protected,
    )


def _branch_match(base: SinglePointResult, refined: SinglePointResult) -> bool | None:
    if base.selected_trap_index is None and refined.selected_trap_index is None:
        return True
    if base.selected_trap_index is None or refined.selected_trap_index is None:
        return False
    base_d = base.trap_candidates[base.selected_trap_index].minimum.d_m
    refined_stable = [candidate for candidate in refined.trap_candidates if candidate.stable]
    if not refined_stable:
        return False
    nearest = min(refined_stable, key=lambda candidate: abs(np.log(candidate.minimum.d_m / base_d)))
    return nearest.candidate_index == refined.selected_trap_index


def run_convergence_check(
    point: SweepPointInput,
    analysis_config: TrapAnalysisConfig,
    base: SinglePointResult,
    *,
    energy_evaluator: EnergyEvaluator,
) -> SinglePointResult:
    """Run a new real refined grid and apply classification/position/depth checks."""
    from .analysis import _analyze_pass

    started = time.perf_counter()
    refined_config = replace(
        analysis_config,
        initial_grid_points=analysis_config.initial_grid_points
        * analysis_config.convergence.refined_grid_factor,
        convergence=replace(analysis_config.convergence, enabled=False),
    )
    refined = _analyze_pass(point, refined_config, energy_evaluator=energy_evaluator)
    d_change = _safe_relative_change(base.d_trap_m, refined.d_trap_m)
    delta_change = _safe_relative_change(base.delta_u_native, refined.delta_u_native)
    stable_count_match = base.stable_trap_count == refined.stable_trap_count
    branch_match = _branch_match(base, refined)
    classification_match = (
        base.trap_exists == refined.trap_exists
        and (
            (base.status == refined.status)
            or ({base.status, refined.status} <= {"success", "multiple_traps"})
        )
    )
    barrier_type_match = _selected_barrier_signature(base) == _selected_barrier_signature(refined)
    cutoff_match = base.cutoff_sensitive == refined.cutoff_sensitive
    position_ok = d_change is None or d_change <= analysis_config.convergence.d_trap_rtol
    depth_ok = delta_change is None or delta_change <= analysis_config.convergence.delta_u_rtol
    if base.delta_u_infinite != refined.delta_u_infinite:
        depth_ok = False
    passed = bool(
        base.status != "numerical_failure"
        and refined.status != "numerical_failure"
        and classification_match
        and barrier_type_match
        and cutoff_match
        and position_ok
        and depth_ok
        and (
            stable_count_match
            or not analysis_config.convergence.require_same_stable_trap_count
        )
        and (branch_match or not analysis_config.convergence.require_same_selected_branch)
    )
    warnings: list[str] = []
    if not stable_count_match:
        warnings.append("stable trap count changed on the independently refined grid")
    if not branch_match:
        warnings.append("selected branch could not be matched by nearest log-distance")
    if not classification_match:
        warnings.append("trap_exists or top-level classification changed on the refined grid")
    if not barrier_type_match:
        warnings.append("selected near/far barrier type or protection changed")
    if not cutoff_match:
        warnings.append("cutoff_sensitive changed on the refined grid")
    if not position_ok:
        warnings.append("selected d_trap exceeds configured relative convergence tolerance")
    if not depth_ok:
        warnings.append("selected delta_u exceeds configured relative convergence tolerance")
    convergence = ConvergenceResult(
        passed=passed,
        base_grid_points=len(base.distances_m),
        refined_grid_points=len(refined.distances_m),
        base_d_trap_m=base.d_trap_m,
        refined_d_trap_m=refined.d_trap_m,
        d_trap_relative_change=d_change,
        base_delta_u_native=base.delta_u_native,
        refined_delta_u_native=refined.delta_u_native,
        delta_u_relative_change=delta_change,
        stable_count_match=stable_count_match,
        branch_match=branch_match,
        classification_match=classification_match,
        barrier_type_match=barrier_type_match,
        cutoff_sensitive_match=cutoff_match,
        base_evaluation_count=base.backend_evaluation_count,
        refined_evaluation_count=refined.backend_evaluation_count,
        runtime_s=time.perf_counter() - started,
        warnings=warnings,
    )
    combined_metadata = {
        "base": base.convergence_metadata,
        "refined": refined.convergence_metadata,
        "actual_separation_evaluations": base.backend_evaluation_count
        + refined.backend_evaluation_count,
    }
    total_runtime = (base.runtime_s or 0.0) + (refined.runtime_s or 0.0)
    if passed:
        return replace(
            refined,
            convergence=convergence,
            convergence_metadata=combined_metadata,
            runtime_s=total_runtime,
            publication_ready=refined.publication_ready,
            warnings=[*refined.warnings, *warnings],
        )
    return replace(
        base,
        status="not_converged",
        trap_exists=None,
        selected_trap_index=None,
        selected_trap_rule=None,
        d_trap_m=None,
        delta_u_native=None,
        delta_u_reporting=None,
        delta_u_infinite=False,
        publication_ready=False,
        convergence=convergence,
        convergence_metadata=combined_metadata,
        trap_classification="not_converged",
        delta_U=None,
        near_barrier=None,
        far_barrier=None,
        warnings=[*base.warnings, *warnings],
        runtime_s=total_runtime,
    )


__all__ = ["run_convergence_check"]
