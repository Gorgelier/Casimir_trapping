"""Serializable data models for sweep points, energy curves, and trap analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
RESULT_STATUSES = frozenset(
    {
        "success",
        "multiple_traps",
        "no_trap",
        "boundary_minimum",
        "unresolved_tail",
        "not_converged",
        "invalid_geometry",
        "unresolved_configuration",
        "numerical_failure",
        "numerically_unresolved",
        "not_evaluated",
    }
)
STABILITY_CLASSES = frozenset(
    {
        "stable_finite_barriers",
        "stable_monotonic_far_confinement",
        "stable_cutoff_limited",
        "no_trap_open_contact",
        "no_trap_open_far",
        "no_trap_unresolved_shallow_minimum",
        "boundary_minimum",
        "unresolved_tail",
        "numerically_unresolved",
    }
)
BARRIER_KINDS = frozenset(
    {
        "interior_maximum",
        "cutoff_boundary",
        "asymptotic_plateau",
        "monotonic_confinement",
        "open_contact_escape",
        "open_far_escape",
        "unresolved_boundary",
    }
)
RUN_STATES = frozenset({"planned", "running", "partial", "completed", "interrupted", "blocked", "failed_validation"})
POINT_EXECUTION_STATUSES = frozenset(
    {
        "pending",
        "running",
        "completed",
        "skipped_valid",
        "timed_out",
        "worker_crash",
        "interrupted",
        "corrupt_existing",
        "stale_existing",
        "incomplete_existing",
        "blocked",
    }
)
RESULT_QUALITIES = frozenset(
    {
        "diagnostic_smoke",
        "diagnostic_coarse",
        "production_resolved",
        "production_cutoff_blocked",
        "production_not_converged",
        "production_unresolved",
        "execution_failed",
        "pending",
    }
)
_NO_TOP_LEVEL_TRAP_STATUSES = frozenset(
    {
        "no_trap",
        "boundary_minimum",
        "unresolved_tail",
        "not_converged",
        "invalid_geometry",
        "unresolved_configuration",
        "numerical_failure",
        "numerically_unresolved",
        "not_evaluated",
    }
)


def _array(values: object, field_name: str) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{field_name} must be a one-dimensional array.")
    return result


@dataclass(frozen=True)
class AdaptiveExtensionConfig:
    enabled: bool
    extension_factor: float
    max_extensions: int
    edge_fraction: float
    tail_points: int


@dataclass(frozen=True)
class ExtremaRefinementConfig:
    enabled: bool
    method: str
    xatol: float
    max_evaluations: int
    merge_distance_rtol: float


@dataclass(frozen=True)
class NumericalResolutionConfig:
    energy_comparison_rtol: float
    slope_zero_rtol: float
    tail_flatness_rtol: float
    tail_extension_rtol: float
    gravity_dominance_ratio: float
    energy_atol_native: float


@dataclass(frozen=True)
class SelectionConfig:
    primary_rule: str
    tie_break_rule: str


@dataclass(frozen=True)
class ConvergenceConfig:
    enabled: bool
    refined_grid_factor: int
    d_trap_rtol: float
    delta_u_rtol: float
    require_same_stable_trap_count: bool
    require_same_selected_branch: bool


@dataclass(frozen=True)
class TrapAnalysisConfig:
    profile: Literal["smoke", "coarse", "production"]
    coordinate: str
    initial_grid_points: int
    adaptive_extension: AdaptiveExtensionConfig
    extrema_refinement: ExtremaRefinementConfig
    numerical_resolution: NumericalResolutionConfig
    selection: SelectionConfig
    convergence: ConvergenceConfig
    max_runtime_seconds: float | None = None


@dataclass(frozen=True)
class SweepExecutionConfig:
    execution_mode: str
    workers: int
    default_max_selected_points: int
    require_execute_model_flag: bool
    require_full_series_confirmation: bool
    require_production_confirmation: bool
    prohibit_run_all: bool
    timeout_seconds: dict[str, float]
    total_runtime_budget_seconds: dict[str, float | None]
    default_retries: int
    maximum_retries: int
    retryable_execution_statuses: tuple[str, ...]
    atomic_point_writes: bool
    compressed_npz: bool
    checksum_algorithm: str
    completed_marker: str
    minimum_free_disk_gb: float
    verify_checksums: bool
    verify_point_hash: bool
    verify_config_hash: bool
    verify_source_fingerprint: bool
    skip_valid_completed_points: bool
    refuse_stale_results: bool
    refuse_corrupt_results: bool
    save_worker_stdout: bool
    save_worker_stderr: bool
    events_jsonl: bool


@dataclass(frozen=True)
class SweepPointInput:
    geometry: str
    gravity_case: str
    t_Au_m: float | None
    t_PTFE_m: float
    R_cyl_m: float | None
    R_sph_m: float | None
    temperature_K: float
    gravity_parameters: dict[str, Any]
    distance_grid_settings: dict[str, Any]
    model_version: str
    config_hash: str
    physical_parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SweepPointInput":
        return cls(**value)


@dataclass(frozen=True)
class EnergyCurveResult:
    geometry: str
    gravity_case: str
    distances_m: FloatArray
    interaction_energy: FloatArray
    gravity_energy: FloatArray
    total_energy: FloatArray
    energy_unit: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        distances = _array(self.distances_m, "distances_m")
        interaction = _array(self.interaction_energy, "interaction_energy")
        gravity = _array(self.gravity_energy, "gravity_energy")
        total = _array(self.total_energy, "total_energy")
        expected = distances.shape
        for name, values in (
            ("interaction_energy", interaction),
            ("gravity_energy", gravity),
            ("total_energy", total),
        ):
            if values.shape != expected:
                raise ValueError(f"{name} shape {values.shape} does not match distances_m shape {expected}.")
        if distances.size == 0 or not np.all(np.isfinite(distances)) or np.any(distances <= 0.0):
            raise ValueError("distances_m must contain positive finite values.")
        if distances.size > 1 and np.any(np.diff(distances) <= 0.0):
            raise ValueError("distances_m must be strictly increasing.")
        for name, values in (
            ("interaction_energy", interaction),
            ("gravity_energy", gravity),
            ("total_energy", total),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")
        object.__setattr__(self, "distances_m", distances)
        object.__setattr__(self, "interaction_energy", interaction)
        object.__setattr__(self, "gravity_energy", gravity)
        object.__setattr__(self, "total_energy", total)


@dataclass(frozen=True)
class StationaryPoint:
    kind: Literal["minimum", "maximum"]
    d_m: float
    x: float
    interaction_energy_native: float
    gravity_energy_native: float
    total_energy_native: float
    bracket_d_m: tuple[float, float]
    left_slope: float
    right_slope: float
    curvature_x: float
    refinement_method: str
    refinement_success: bool
    evaluation_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StationaryPoint":
        data = dict(value)
        data["bracket_d_m"] = tuple(data["bracket_d_m"])
        return cls(**data)


@dataclass(frozen=True)
class EscapeBarrier:
    side: Literal["near", "far"]
    kind: str
    location_d_m: float | None
    energy_native: float | None
    height_native: float | None
    height_reporting: float | None
    protected: bool
    finite: bool
    is_infinite: bool = False
    lower_bound: bool = False
    unresolved: bool = False
    uncertainty_native: float | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in BARRIER_KINDS:
            raise ValueError(f"Unsupported barrier kind {self.kind!r}.")
        if self.is_infinite and (self.finite or self.height_native is not None):
            raise ValueError("An infinite barrier must use finite=False and height_native=None.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EscapeBarrier":
        return cls(**value)


@dataclass(frozen=True)
class TrapCandidate:
    candidate_index: int
    minimum: StationaryPoint
    near_barrier: EscapeBarrier
    far_barrier: EscapeBarrier
    stable: bool
    stability_class: str
    delta_u_native: float | None
    delta_u_reporting: float | None
    delta_u_is_lower_bound: bool
    delta_u_infinite: bool
    cutoff_sensitive: bool
    publication_ready: bool
    branch_order: int
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.stability_class not in STABILITY_CLASSES:
            raise ValueError(f"Unsupported stability class {self.stability_class!r}.")
        if not self.stable and (self.delta_u_native is not None or self.delta_u_infinite):
            raise ValueError("An unstable trap candidate cannot expose a trap depth.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrapCandidate":
        data = dict(value)
        data["minimum"] = StationaryPoint.from_dict(data["minimum"])
        data["near_barrier"] = EscapeBarrier.from_dict(data["near_barrier"])
        data["far_barrier"] = EscapeBarrier.from_dict(data["far_barrier"])
        return cls(**data)


@dataclass(frozen=True)
class DistanceExtension:
    extension_index: int
    previous_d_max_m: float
    new_d_max_m: float
    reasons: list[str]
    new_point_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConvergenceResult:
    passed: bool | None
    base_grid_points: int
    refined_grid_points: int
    base_d_trap_m: float | None
    refined_d_trap_m: float | None
    d_trap_relative_change: float | None
    base_delta_u_native: float | None
    refined_delta_u_native: float | None
    delta_u_relative_change: float | None
    stable_count_match: bool | None
    branch_match: bool | None
    classification_match: bool | None
    barrier_type_match: bool | None
    cutoff_sensitive_match: bool | None
    base_evaluation_count: int
    refined_evaluation_count: int
    runtime_s: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConvergenceResult":
        return cls(**value)


@dataclass(frozen=True)
class SinglePointResult:
    status: str
    geometry: str
    gravity_case: str
    input_parameters: dict[str, Any]
    distances_m: FloatArray = field(default_factory=lambda: np.array([], dtype=float))
    interaction_energy: FloatArray = field(default_factory=lambda: np.array([], dtype=float))
    gravity_energy: FloatArray | None = None
    total_energy: FloatArray = field(default_factory=lambda: np.array([], dtype=float))
    interaction_energy_reporting: FloatArray = field(default_factory=lambda: np.array([], dtype=float))
    gravity_energy_reporting: FloatArray = field(default_factory=lambda: np.array([], dtype=float))
    total_energy_reporting: FloatArray = field(default_factory=lambda: np.array([], dtype=float))
    energy_unit: str | None = None
    reporting_unit: str | None = None
    trap_exists: bool | None = None
    stationary_points: list[StationaryPoint] = field(default_factory=list)
    local_minimum_count: int = 0
    stable_trap_count: int = 0
    trap_candidates: list[TrapCandidate] = field(default_factory=list)
    selected_trap_index: int | None = None
    selected_trap_rule: str | None = None
    global_energy_minimum_index: int | None = None
    largest_barrier_index: int | None = None
    d_trap_m: float | None = None
    delta_u_native: float | None = None
    delta_u_reporting: float | None = None
    delta_u_infinite: bool = False
    cutoff_sensitive: bool = False
    publication_ready: bool = False
    distance_extensions: list[DistanceExtension] = field(default_factory=list)
    convergence: ConvergenceResult | None = None
    # Compatibility fields retained from batch 1.
    trap_classification: str = "not_evaluated"
    trap_count: int | None = None
    delta_U: float | None = None
    near_barrier: float | None = None
    far_barrier: float | None = None
    warnings: list[str] = field(default_factory=list)
    convergence_metadata: dict[str, Any] = field(default_factory=dict)
    runtime_s: float | None = None
    exception_information: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"Unsupported result status {self.status!r}.")
        if self.status in _NO_TOP_LEVEL_TRAP_STATUSES:
            if self.d_trap_m is not None or self.delta_u_native is not None or self.delta_U is not None:
                raise ValueError(
                    f"status={self.status!r} requires d_trap_m=None, delta_u_native=None, and delta_U=None."
                )
        if self.status in {"success", "multiple_traps"} and self.trap_exists is False:
            raise ValueError(f"status={self.status!r} cannot use trap_exists=False.")
        distances = _array(self.distances_m, "distances_m")
        interaction = _array(self.interaction_energy, "interaction_energy")
        total = _array(self.total_energy, "total_energy")
        gravity = None if self.gravity_energy is None else _array(self.gravity_energy, "gravity_energy")
        interaction_reporting = _array(self.interaction_energy_reporting, "interaction_energy_reporting")
        gravity_reporting = _array(self.gravity_energy_reporting, "gravity_energy_reporting")
        total_reporting = _array(self.total_energy_reporting, "total_energy_reporting")
        native_lengths = {distances.size, interaction.size, total.size}
        if gravity is not None:
            native_lengths.add(gravity.size)
        if len(native_lengths) != 1:
            raise ValueError("All populated native energy arrays must have the same length as distances_m.")
        reporting_lengths = {interaction_reporting.size, gravity_reporting.size, total_reporting.size}
        if reporting_lengths != {0} and reporting_lengths != {distances.size}:
            raise ValueError("All reporting-energy arrays must be empty or match distances_m.")
        object.__setattr__(self, "distances_m", distances)
        object.__setattr__(self, "interaction_energy", interaction)
        object.__setattr__(self, "total_energy", total)
        object.__setattr__(self, "gravity_energy", gravity)
        object.__setattr__(self, "interaction_energy_reporting", interaction_reporting)
        object.__setattr__(self, "gravity_energy_reporting", gravity_reporting)
        object.__setattr__(self, "total_energy_reporting", total_reporting)

    @property
    def backend_evaluation_count(self) -> int:
        value = self.convergence_metadata.get("actual_separation_evaluations", 0)
        return int(value) if isinstance(value, (int, float)) else 0

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status,
            "geometry": self.geometry,
            "gravity_case": self.gravity_case,
            "input_parameters": self.input_parameters,
            "energy_unit": self.energy_unit,
            "reporting_unit": self.reporting_unit,
            "trap_exists": self.trap_exists,
            "stationary_points": [point.to_dict() for point in self.stationary_points],
            "local_minimum_count": self.local_minimum_count,
            "stable_trap_count": self.stable_trap_count,
            "trap_candidates": [candidate.to_dict() for candidate in self.trap_candidates],
            "selected_trap_index": self.selected_trap_index,
            "selected_trap_rule": self.selected_trap_rule,
            "global_energy_minimum_index": self.global_energy_minimum_index,
            "largest_barrier_index": self.largest_barrier_index,
            "d_trap_m": self.d_trap_m,
            "delta_u_native": self.delta_u_native,
            "delta_u_reporting": self.delta_u_reporting,
            "delta_u_infinite": self.delta_u_infinite,
            "cutoff_sensitive": self.cutoff_sensitive,
            "publication_ready": self.publication_ready,
            "distance_extensions": [extension.to_dict() for extension in self.distance_extensions],
            "convergence": None if self.convergence is None else self.convergence.to_dict(),
            "trap_classification": self.trap_classification,
            "trap_count": self.trap_count,
            "delta_U": self.delta_U,
            "near_barrier": self.near_barrier,
            "far_barrier": self.far_barrier,
            "warnings": self.warnings,
            "convergence_metadata": self.convergence_metadata,
            "runtime_s": self.runtime_s,
            "exception_information": self.exception_information,
        }
        if include_arrays:
            data.update(
                {
                    "d_m": self.distances_m.tolist(),
                    "interaction_energy": self.interaction_energy.tolist(),
                    "gravity_energy": None if self.gravity_energy is None else self.gravity_energy.tolist(),
                    "total_energy": self.total_energy.tolist(),
                    "interaction_energy_reporting": self.interaction_energy_reporting.tolist(),
                    "gravity_energy_reporting": self.gravity_energy_reporting.tolist(),
                    "total_energy_reporting": self.total_energy_reporting.tolist(),
                }
            )
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SinglePointResult":
        data = dict(value)
        data["distances_m"] = data.pop("d_m", [])
        data.setdefault("interaction_energy", [])
        data.setdefault("gravity_energy", None)
        data.setdefault("total_energy", [])
        data.setdefault("interaction_energy_reporting", [])
        data.setdefault("gravity_energy_reporting", [])
        data.setdefault("total_energy_reporting", [])
        data["stationary_points"] = [StationaryPoint.from_dict(item) for item in data.get("stationary_points", [])]
        data["trap_candidates"] = [TrapCandidate.from_dict(item) for item in data.get("trap_candidates", [])]
        data["distance_extensions"] = [DistanceExtension(**item) for item in data.get("distance_extensions", [])]
        if data.get("convergence") is not None:
            data["convergence"] = ConvergenceResult.from_dict(data["convergence"])
        return cls(**data)


@dataclass(frozen=True)
class SeriesPointSpec:
    series_id: str
    geometry: str
    gravity_case: str
    grid_profile: str
    analysis_profile: str
    series_index: int
    total_series_points: int
    swept_parameter_name: str
    swept_parameter_value_si: float
    fixed_parameters_si: dict[str, float]
    sweep_point_input: SweepPointInput
    point_hash: str
    point_id: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sweep_point_input"] = self.sweep_point_input.to_dict()
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SeriesPointSpec":
        data = dict(value)
        data["sweep_point_input"] = SweepPointInput.from_dict(data["sweep_point_input"])
        return cls(**data)


@dataclass(frozen=True)
class SeriesRunPlan:
    schema_version: int
    run_id: str
    series_id: str
    geometry: str
    gravity_case: str
    grid_profile: str
    analysis_profile: str
    check_convergence: bool
    full_series_point_count: int
    selected_point_count: int
    selected_indices: tuple[int, ...]
    points: tuple[SeriesPointSpec, ...]
    config_path: str
    config_hash: str
    analysis_config_hash: str
    parameter_grid_hash: str
    source_fingerprint: str
    creation_time_utc: str
    selection_method: str
    diagnostic_only: bool
    production_eligible: bool
    production_validation_passed: bool
    blockers: tuple[str, ...]
    plan_hash: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["selected_indices"] = list(self.selected_indices)
        data["points"] = [point.to_dict() for point in self.points]
        data["blockers"] = list(self.blockers)
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SeriesRunPlan":
        data = dict(value)
        data["selected_indices"] = tuple(data["selected_indices"])
        data["points"] = tuple(SeriesPointSpec.from_dict(point) for point in data["points"])
        data["blockers"] = tuple(data.get("blockers", []))
        return cls(**data)


@dataclass
class PointExecutionRecord:
    point_id: str
    series_index: int
    execution_status: str = "pending"
    analysis_status: str | None = None
    attempt_count: int = 0
    start_time_utc: str | None = None
    end_time_utc: str | None = None
    runtime_seconds: float | None = None
    worker_exit_code: int | None = None
    timed_out: bool = False
    result_directory: str | None = None
    metadata_checksum: str | None = None
    npz_checksum: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    warning_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PointExecutionRecord":
        return cls(**value)


@dataclass
class SeriesRunManifest:
    run_id: str
    plan_hash: str
    state: str
    planned_points: int
    completed_points: int
    failed_points: int
    timed_out_points: int
    pending_points: int
    skipped_valid_points: int
    stale_points: int
    corrupt_points: int
    start_time_utc: str | None
    last_update_time_utc: str
    end_time_utc: str | None
    elapsed_seconds: float
    execution_records: list[PointExecutionRecord]
    retry_analysis_failures: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["execution_records"] = [record.to_dict() for record in self.execution_records]
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SeriesRunManifest":
        data = dict(value)
        data["execution_records"] = [
            PointExecutionRecord.from_dict(record) for record in data.get("execution_records", [])
        ]
        return cls(**data)


__all__ = [
    "AdaptiveExtensionConfig",
    "BARRIER_KINDS",
    "ConvergenceConfig",
    "ConvergenceResult",
    "DistanceExtension",
    "EnergyCurveResult",
    "EscapeBarrier",
    "ExtremaRefinementConfig",
    "NumericalResolutionConfig",
    "POINT_EXECUTION_STATUSES",
    "RESULT_STATUSES",
    "RESULT_QUALITIES",
    "RUN_STATES",
    "STABILITY_CLASSES",
    "SelectionConfig",
    "SeriesPointSpec",
    "SeriesRunManifest",
    "SeriesRunPlan",
    "SinglePointResult",
    "StationaryPoint",
    "SweepPointInput",
    "SweepExecutionConfig",
    "TrapAnalysisConfig",
    "TrapCandidate",
    "PointExecutionRecord",
]
