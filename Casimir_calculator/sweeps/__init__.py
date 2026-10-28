"""Numerical energy curves and strict trap analysis for focused size scans."""

from .adapter import evaluate_energy_curve
from .analysis import CachedEnergyEvaluator, analyze_single_point, reporting_conversion
from .grids import build_distance_grid, build_grid, distance_max_m
from .models import (
    AdaptiveExtensionConfig, ConvergenceConfig, ConvergenceResult,
    EnergyCurveResult, EscapeBarrier, ExtremaRefinementConfig,
    NumericalResolutionConfig, SelectionConfig, SinglePointResult,
    StationaryPoint, SweepPointInput, TrapAnalysisConfig, TrapCandidate,
)

__all__ = [
    "AdaptiveExtensionConfig", "CachedEnergyEvaluator", "ConvergenceConfig",
    "ConvergenceResult", "EnergyCurveResult", "EscapeBarrier",
    "ExtremaRefinementConfig", "NumericalResolutionConfig", "SelectionConfig",
    "SinglePointResult", "StationaryPoint", "SweepPointInput",
    "TrapAnalysisConfig", "TrapCandidate", "analyze_single_point",
    "build_distance_grid", "build_grid", "distance_max_m",
    "evaluate_energy_curve", "reporting_conversion",
]
