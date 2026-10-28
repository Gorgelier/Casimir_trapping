"""Central SI unit conversions used by the sweep framework."""

from __future__ import annotations

import math


METER = 1.0
NANOMETER = 1.0e-9
MICROMETER = 1.0e-6
KELVIN = 1.0
JOULE = 1.0
NEWTON = 1.0


def require_finite_si(value: object, field_path: str) -> float:
    """Return one finite numeric SI value or raise a contextual error."""
    if isinstance(value, bool):
        raise ValueError(f"{field_path}: current value={value!r}; expected a finite SI number; replace it with a numeric value.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field_path}: current value={value!r}; expected a finite SI number; "
            "replace it with a numeric value expressed in the declared SI unit."
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError(
            f"{field_path}: current value={value!r}; expected a finite SI number; "
            "remove NaN or infinity."
        )
    return numeric


__all__ = [
    "JOULE",
    "KELVIN",
    "METER",
    "MICROMETER",
    "NANOMETER",
    "NEWTON",
    "require_finite_si",
]
