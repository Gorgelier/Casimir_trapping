from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .constants import BOLTZMANN, C, HBAR
from .psd import psd

FrequencyFunction = Callable[[float], np.ndarray]

_MSD_TAIL_WINDOW = 5
_MSD_MAX_TERMS = 100_000


def _as_vector(value: np.ndarray | tuple[float, ...] | list[float]) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _infinity_norm(value: np.ndarray) -> float:
    flattened = np.ravel(value)
    if flattened.size == 0:
        return 0.0
    return float(np.max(np.abs(flattened)))


def _validated_order(order: int | None) -> int | None:
    if order is None:
        return None
    if isinstance(order, (bool, np.bool_)):
        raise ValueError("order must be a positive integer or None.")
    try:
        integer_order = int(order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("order must be a positive integer or None.") from exc
    if integer_order < 1 or integer_order != order:
        raise ValueError("order must be a positive integer or None.")
    return integer_order


def _validate_inputs(temperature_k: float, distance_m: float, epsrel: float) -> None:
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("Frequency summation requires positive finite temperature.")
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("distance_m must be finite and positive.")
    if not np.isfinite(epsrel) or epsrel <= 0.0:
        raise ValueError("epsrel must be finite and positive.")


def automatic_psd_order(temperature_k: float, distance_m: float, epsrel: float) -> int:
    """Match the PSD order heuristic used in califorcia/casimir_cylinder."""
    _validate_inputs(temperature_k, distance_m, epsrel)
    teff = 4.0 * np.pi * BOLTZMANN * temperature_k * distance_m / (HBAR * C)
    return int(max(np.ceil((1.0 - 1.5 * np.log10(epsrel)) / np.sqrt(teff)), 5))


def psd_sum(
    temperature_k: float,
    distance_m: float,
    func: FrequencyFunction,
    epsrel: float,
    order: int | None = None,
) -> np.ndarray:
    """Finite-frequency contribution using Pade spectrum decomposition.

    `func` is evaluated at vacuum wave number K = xi/c and returns a vector of
    polarization or observable contributions. This mirrors the interface used
    by califorcia and casimir_cylinder.
    """
    _validate_inputs(temperature_k, distance_m, epsrel)
    order = _validated_order(order)
    if order is None:
        order = automatic_psd_order(temperature_k, distance_m, epsrel)

    matsubara_k = 2.0 * np.pi * BOLTZMANN * temperature_k / (HBAR * C)
    poles, weights = psd(order)
    result = None
    for pole, weight in zip(poles, weights):
        term = 2.0 * weight * _as_vector(func(matsubara_k * pole / (2.0 * np.pi)))
        result = term if result is None else result + term
    if result is None:
        result = np.zeros(2)
    return 0.5 * BOLTZMANN * temperature_k * result


def msd_sum(
    temperature_k: float,
    distance_m: float,
    func: FrequencyFunction,
    epsrel: float,
    order: int | None = None,
) -> np.ndarray:
    """Finite-frequency contribution using direct Matsubara summation."""
    _validate_inputs(temperature_k, distance_m, epsrel)
    order = _validated_order(order)

    matsubara_k = 2.0 * np.pi * BOLTZMANN * temperature_k / (HBAR * C)
    max_index = _MSD_MAX_TERMS if order is None else order
    result = None
    recent_norms: list[float] = []
    n = 1
    while n <= max_index:
        term = 2.0 * _as_vector(func(matsubara_k * n))
        if not np.all(np.isfinite(term)):
            raise FloatingPointError(f"MSD term n={n} is non-finite.")
        result = term if result is None else result + term
        if order is None:
            recent_norms.append(_infinity_norm(term))
            if len(recent_norms) > _MSD_TAIL_WINDOW:
                recent_norms.pop(0)

            if len(recent_norms) == _MSD_TAIL_WINDOW:
                # Five exact zero terms are safe for functions with compact
                # support and also avoid stopping at one accidental zero.
                if all(term_norm == 0.0 for term_norm in recent_norms):
                    break

                result_scale = max(_infinity_norm(result), np.finfo(float).tiny)
                all_small = all(
                    term_norm <= epsrel * result_scale
                    for term_norm in recent_norms
                )
                # Estimate the uncomputed tail only after a run of strictly
                # decreasing, nonzero terms.  The largest recent ratio gives
                # a conservative geometric bound for the asymptotic tail.
                if all_small and all(term_norm > 0.0 for term_norm in recent_norms):
                    ratios = [
                        current / previous
                        for previous, current in zip(recent_norms[:-1], recent_norms[1:])
                    ]
                    if all(ratio < 1.0 for ratio in ratios):
                        ratio_bound = max(ratios)
                        tail_bound = recent_norms[-1] * ratio_bound / (1.0 - ratio_bound)
                        if tail_bound <= epsrel * result_scale:
                            break
        n += 1
    else:
        if order is None:
            raise RuntimeError(
                "Automatic MSD summation did not converge within "
                f"{_MSD_MAX_TERMS} nonzero Matsubara terms; use PSD summation "
                "or provide an explicit converged order."
            )
    if result is None:
        result = np.zeros(2)
    return 0.5 * BOLTZMANN * temperature_k * result


__all__ = ["automatic_psd_order", "msd_sum", "psd_sum"]
