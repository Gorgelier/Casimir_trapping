from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import C


def longitudinal_wavevector(
    epsilon: float | np.ndarray,
    xi: float | np.ndarray,
    k_parallel: float | np.ndarray,
) -> np.ndarray:
    epsilon_array = np.asarray(epsilon, dtype=float)
    xi_array = np.asarray(xi, dtype=float)
    k_parallel_array = np.asarray(k_parallel, dtype=float)
    return np.sqrt(k_parallel_array**2 + epsilon_array * (xi_array / C) ** 2)


def _normalize_polarization(polarization: str) -> str:
    key = polarization.upper()
    if key in {"TE", "S"}:
        return "TE"
    if key in {"TM", "P"}:
        return "TM"
    raise ValueError(f"Unsupported polarization: {polarization}")


def fresnel_coefficient(
    epsilon_left: float | np.ndarray,
    epsilon_right: float | np.ndarray,
    kappa_left: float | np.ndarray,
    kappa_right: float | np.ndarray,
    polarization: str,
) -> np.ndarray:
    mode = _normalize_polarization(polarization)
    epsilon_left_array = np.asarray(epsilon_left, dtype=float)
    epsilon_right_array = np.asarray(epsilon_right, dtype=float)
    kappa_left_array = np.asarray(kappa_left, dtype=float)
    kappa_right_array = np.asarray(kappa_right, dtype=float)
    if mode == "TE":
        numerator = kappa_left_array - kappa_right_array
        denominator = kappa_left_array + kappa_right_array
        return _divide_with_limit(numerator, denominator, 0.0)

    numerator = epsilon_right_array * kappa_left_array - epsilon_left_array * kappa_right_array
    denominator = epsilon_right_array * kappa_left_array + epsilon_left_array * kappa_right_array
    static_limit = _divide_with_limit(
        epsilon_right_array - epsilon_left_array,
        epsilon_right_array + epsilon_left_array,
        0.0,
    )
    return _divide_with_limit(numerator, denominator, static_limit)


def _divide_with_limit(
    numerator: float | np.ndarray,
    denominator: float | np.ndarray,
    limit: float | np.ndarray,
) -> np.ndarray:
    numerator_array, denominator_array, limit_array = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
        np.asarray(limit, dtype=float),
    )
    result = limit_array.astype(float, copy=True)
    mask = np.abs(denominator_array) > np.finfo(float).tiny
    np.divide(numerator_array, denominator_array, out=result, where=mask)
    return result


def interface_reflection(
    epsilon_left: float | np.ndarray,
    epsilon_right: float | np.ndarray,
    xi: float | np.ndarray,
    k_parallel: float | np.ndarray,
    polarization: str,
) -> np.ndarray:
    kappa_left = longitudinal_wavevector(epsilon_left, xi, k_parallel)
    kappa_right = longitudinal_wavevector(epsilon_right, xi, k_parallel)
    return fresnel_coefficient(epsilon_left, epsilon_right, kappa_left, kappa_right, polarization)


def layered_reflection(
    epsilons: Sequence[float | np.ndarray],
    thicknesses_m: Sequence[float],
    xi: float | np.ndarray,
    k_parallel: float | np.ndarray,
    polarization: str,
) -> np.ndarray:
    if len(epsilons) < 2:
        raise ValueError("At least two media are required.")
    if len(thicknesses_m) != len(epsilons) - 2:
        raise ValueError("There must be exactly one thickness per internal layer.")

    epsilon_arrays = [np.asarray(epsilon, dtype=float) for epsilon in epsilons]
    kappas = [longitudinal_wavevector(epsilon, xi, k_parallel) for epsilon in epsilon_arrays]

    reflection = fresnel_coefficient(
        epsilon_arrays[-2],
        epsilon_arrays[-1],
        kappas[-2],
        kappas[-1],
        polarization,
    )
    for layer_index in range(len(thicknesses_m) - 1, -1, -1):
        phase = np.exp(-2.0 * kappas[layer_index + 1] * thicknesses_m[layer_index])
        interface = fresnel_coefficient(
            epsilon_arrays[layer_index],
            epsilon_arrays[layer_index + 1],
            kappas[layer_index],
            kappas[layer_index + 1],
            polarization,
        )
        reflection = (interface + reflection * phase) / (1.0 + interface * reflection * phase)
    return reflection


__all__ = [
    "C",
    "fresnel_coefficient",
    "interface_reflection",
    "layered_reflection",
    "longitudinal_wavevector",
]
