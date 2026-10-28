"""Geometry-independent SI grid construction for parameter sweeps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .units import require_finite_si


Profile = Literal["smoke", "coarse", "production"]
FloatArray = NDArray[np.float64]
PROFILES = ("smoke", "coarse", "production")


def _point_count(spec: Mapping[str, Any], profile: Profile, field_path: str) -> int:
    counts = spec.get("points")
    if not isinstance(counts, Mapping) or profile not in counts:
        raise ValueError(
            f"{field_path}.points.{profile}: current value=missing; expected an integer >= 2; "
            "add the profile-specific point count."
        )
    value = counts[profile]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
        raise ValueError(
            f"{field_path}.points.{profile}: current value={value!r}; expected an integer >= 2; "
            "set a valid base grid size."
        )
    return int(value)


def _finite_sequence(values: object, field_path: str) -> list[float]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(
            f"{field_path}: current value={values!r}; expected a list of finite SI values; "
            "use YAML list syntax."
        )
    return [require_finite_si(value, f"{field_path}[{index}]") for index, value in enumerate(values)]


def _finalize_grid(
    base: FloatArray,
    *,
    anchors: list[float],
    endpoints: tuple[float, float],
    field_path: str,
) -> FloatArray:
    lower, upper = endpoints
    for index, anchor in enumerate(anchors):
        if anchor < lower or anchor > upper:
            raise ValueError(
                f"{field_path}.anchors_m[{index}]: current value={anchor!r}; "
                f"allowed range=[{lower!r}, {upper!r}]; move the anchor inside the grid range."
            )
    values = np.asarray([*base, lower, upper, *anchors], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{field_path}: current value contains NaN or infinity; expected finite SI values; "
            "correct the grid bounds or anchors."
        )
    result = np.unique(values)
    result.sort()
    if result.size < 2 or np.any(np.diff(result) <= 0.0):
        raise ValueError(
            f"{field_path}: current value does not form a strictly increasing grid; "
            "expected at least two unique ordered values; correct bounds and anchors."
        )
    return result


def build_grid(spec: Mapping[str, Any], profile: Profile, *, field_path: str = "grid") -> FloatArray:
    """Build a sorted, deduplicated SI grid with exact endpoints and anchors."""
    if profile not in PROFILES:
        raise ValueError(f"profile: current value={profile!r}; allowed={PROFILES}; choose a supported profile.")
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"{field_path}: current value={spec!r}; expected a grid mapping; provide kind, bounds, and points."
        )

    kind = spec.get("kind")
    count = _point_count(spec, profile, field_path)
    anchors = _finite_sequence(spec.get("anchors_m", []), f"{field_path}.anchors_m")

    if kind in {"log", "linear"}:
        lower = require_finite_si(spec.get("min_m"), f"{field_path}.min_m")
        upper = require_finite_si(spec.get("max_m"), f"{field_path}.max_m")
        if lower < 0.0 or upper <= lower:
            raise ValueError(
                f"{field_path}: current bounds=({lower!r}, {upper!r}); expected 0 <= min_m < max_m; "
                "correct the grid bounds."
            )
        if kind == "log" and lower <= 0.0:
            raise ValueError(
                f"{field_path}.min_m: current value={lower!r}; expected > 0 for a log grid; "
                "use explicit_zero_plus_log when an exact zero is required."
            )
        base = (
            np.geomspace(lower, upper, count, dtype=float)
            if kind == "log"
            else np.linspace(lower, upper, count, dtype=float)
        )
        base[0], base[-1] = lower, upper
        return _finalize_grid(base, anchors=anchors, endpoints=(lower, upper), field_path=field_path)

    if kind == "explicit_zero_plus_log":
        lower = require_finite_si(spec.get("positive_min_m"), f"{field_path}.positive_min_m")
        upper = require_finite_si(spec.get("max_m"), f"{field_path}.max_m")
        if lower <= 0.0 or upper <= lower:
            raise ValueError(
                f"{field_path}: current positive bounds=({lower!r}, {upper!r}); "
                "expected 0 < positive_min_m < max_m; correct the positive-thickness bounds."
            )
        positive = np.geomspace(lower, upper, count, dtype=float)
        positive[0], positive[-1] = lower, upper
        result = _finalize_grid(
            np.concatenate((np.array([0.0]), positive)),
            anchors=anchors,
            endpoints=(0.0, upper),
            field_path=field_path,
        )
        if result[0] != 0.0:
            raise AssertionError("explicit_zero_plus_log failed to retain exact zero")
        return result

    raise ValueError(
        f"{field_path}.kind: current value={kind!r}; allowed=['log', 'linear', "
        "'explicit_zero_plus_log']; select a supported grid kind."
    )


def distance_max_m(distance_spec: Mapping[str, Any], t_ptfe_m: float) -> float:
    """Return max(configured floor, configured PTFE-thickness factor)."""
    thickness = require_finite_si(t_ptfe_m, "t_PTFE_m")
    if thickness < 0.0:
        raise ValueError(
            f"t_PTFE_m: current value={thickness!r}; allowed range=[0, infinity); use a non-negative thickness."
        )
    minimum = require_finite_si(distance_spec.get("d_max_floor_m"), "distance_scan.d_max_floor_m")
    factor = require_finite_si(distance_spec.get("t_PTFE_factor"), "distance_scan.t_PTFE_factor")
    if minimum <= 0.0 or factor <= 0.0:
        raise ValueError(
            "distance_scan: d_max_floor_m and t_PTFE_factor must both be finite and positive; "
            "restore the configured max(2 um, 5*t_PTFE) rule."
        )
    return max(minimum, factor * thickness)


def build_distance_grid(
    distance_spec: Mapping[str, Any],
    t_ptfe_m: float,
    profile: Profile,
) -> FloatArray:
    """Build the initial logarithmic surface-separation grid in metres.

    The normal path is a dense log grid from ``d_min_m`` to ``d_max``.  An
    optional sparse-near mode can instead retain explicit near-cutoff anchors
    and start the bulk log grid at ``bulk_log_min_m``.  It preserves the exact
    lower cutoff used by strict barrier analysis while avoiding redundant
    expensive backend evaluations in a smooth near-field interval.
    """
    lower = require_finite_si(distance_spec.get("d_min_m"), "distance_scan.d_min_m")
    upper = distance_max_m(distance_spec, t_ptfe_m)
    if lower <= 0.0 or lower >= upper:
        raise ValueError(
            f"distance_scan: current bounds=({lower!r}, {upper!r}); expected 0 < d_min_m < d_max_m; "
            "correct d_min_m or the d_max rule."
        )
    count = _point_count(distance_spec, profile, "distance_scan")
    bulk_value = distance_spec.get("bulk_log_min_m")
    anchors_value = distance_spec.get("near_anchors_m")
    if bulk_value is None and anchors_value is None:
        result = np.geomspace(lower, upper, count, dtype=float)
        result[0], result[-1] = lower, upper
    elif bulk_value is None or anchors_value is None:
        raise ValueError(
            "distance_scan.bulk_log_min_m and distance_scan.near_anchors_m must be supplied together."
        )
    else:
        bulk_min = require_finite_si(bulk_value, "distance_scan.bulk_log_min_m")
        if bulk_min < lower or bulk_min >= upper:
            raise ValueError(
                f"distance_scan.bulk_log_min_m: current value={bulk_min!r}; expected within "
                f"[{lower!r}, {upper!r}); correct the bulk log-grid lower bound."
            )
        anchors = _finite_sequence(anchors_value, "distance_scan.near_anchors_m")
        for index, anchor in enumerate(anchors):
            if anchor < lower or anchor > bulk_min:
                raise ValueError(
                    f"distance_scan.near_anchors_m[{index}]: current value={anchor!r}; expected within "
                    f"[{lower!r}, {bulk_min!r}]; correct the sparse near-field anchors."
                )
        extras = np.unique(np.asarray([lower, *anchors], dtype=float))
        endpoint_tolerance = np.finfo(float).eps * max(abs(bulk_min), abs(upper), 1.0) * 16.0
        extras = extras[
            ~np.isclose(extras, bulk_min, rtol=0.0, atol=endpoint_tolerance)
            & ~np.isclose(extras, upper, rtol=0.0, atol=endpoint_tolerance)
        ]
        base_count = count - len(extras)
        if base_count < 2:
            raise ValueError(
                "distance_scan sparse near-field anchors leave fewer than two bulk log-grid points; "
                "increase points or reduce anchors."
            )
        bulk = np.geomspace(bulk_min, upper, base_count, dtype=float)
        bulk[0], bulk[-1] = bulk_min, upper
        result = np.unique(np.concatenate((extras, bulk)))
        result.sort()
        if result.size != count:
            raise ValueError(
                "distance_scan sparse near-field anchors overlap the bulk log grid; choose anchors that "
                "do not duplicate interior bulk points."
            )
    if np.any(np.diff(result) <= 0.0):
        raise ValueError("distance_scan generated a non-increasing grid; correct its bounds and point count.")
    return result


__all__ = [
    "FloatArray",
    "PROFILES",
    "Profile",
    "build_distance_grid",
    "build_grid",
    "distance_max_m",
]
