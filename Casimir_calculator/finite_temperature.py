from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import integrate

from .constants import BOLTZMANN, C, HBAR
from .frequency_summation import msd_sum, psd_sum
from .optics import layered_reflection, longitudinal_wavevector
from .plane import LayeredPlane, PermittivityFunction


_VALID_OBSERVABLES = frozenset({"energy", "pressure", "pressuregradient"})
_OBSERVABLE_DISTANCE_POWER = {"energy": 2, "pressure": 3, "pressuregradient": 4}
_MIN_RELATIVE_TOLERANCE = max(5.0e-29, 50.0 * np.finfo(float).eps)
_ROUNDOFF_TOLERANCE_FACTOR = 256.0


def _positive_integer_or_none(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        integer_value = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if integer_value < 1 or integer_value != value:
        raise ValueError(f"{name} must be a positive integer.")
    return integer_value


def _validate_tolerances(epsrel: float, epsabs: float) -> None:
    if not np.isfinite(epsrel) or epsrel <= 0.0:
        raise ValueError("epsrel must be finite and positive.")
    if not np.isfinite(epsabs) or epsabs < 0.0:
        raise ValueError("epsabs must be finite and non-negative.")
    if epsabs == 0.0 and epsrel <= _MIN_RELATIVE_TOLERANCE:
        raise ValueError(
            "When epsabs is zero, epsrel must be greater than "
            f"{_MIN_RELATIVE_TOLERANCE:.3e} for scipy.integrate."
        )


def _effective_epsabs(epsabs: float, distance_m: float, distance_power: int) -> float:
    """Add a roundoff floor at the observable's natural inverse-length scale."""
    natural_scale = distance_m ** (-distance_power)
    if not np.isfinite(natural_scale):
        raise ValueError("distance_m is too small for finite-precision integration.")
    roundoff_floor = _ROUNDOFF_TOLERANCE_FACTOR * np.finfo(float).eps * natural_scale
    return max(epsabs, float(roundoff_floor))


@dataclass(frozen=True)
class FiniteTemperatureConfig:
    """Finite-temperature summation and SciPy integration controls.

    The calculation is organized as an ``n = 0`` half-weight term followed by
    a finite-frequency PSD or MSD sum. ``frequency_order`` controls that sum.
    ``xi_nodes`` and ``k_nodes`` remain accepted for older project scripts.
    ``xi_nodes`` is a legacy alias for ``frequency_order``; supplying different
    values for both is an error rather than silently changing the requested
    order. ``k_nodes`` is retained only for API compatibility because the
    radial wave-vector integral is adaptive.
    """

    temperature_k: float = 300.0
    frequency_order: int | None = None
    fs: str = "psd"
    epsrel: float = 1.0e-8
    epsabs: float = 0.0
    xi_nodes: int | None = None
    k_nodes: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature_k) or self.temperature_k < 0.0:
            raise ValueError("temperature_k must be finite and non-negative.")
        if self.fs not in {"psd", "msd"}:
            raise ValueError("fs must be 'psd' or 'msd'.")
        _validate_tolerances(self.epsrel, self.epsabs)

        order = _positive_integer_or_none("frequency_order", self.frequency_order)
        legacy_order = _positive_integer_or_none("xi_nodes", self.xi_nodes)
        legacy_k_nodes = _positive_integer_or_none("k_nodes", self.k_nodes)
        if order is not None and legacy_order is not None and order != legacy_order:
            raise ValueError(
                "frequency_order and its legacy alias xi_nodes must match when both are set."
            )
        if order is None and legacy_order is not None:
            order = legacy_order

        object.__setattr__(self, "frequency_order", order)
        object.__setattr__(self, "xi_nodes", legacy_order)
        object.__setattr__(self, "k_nodes", legacy_k_nodes)


def matsubara_frequencies(config: FiniteTemperatureConfig) -> np.ndarray:
    """Return ``xi_n`` for ``n = 0`` through the configured nonzero order."""
    order = config.frequency_order if config.frequency_order is not None else 40
    n_values = np.arange(order + 1, dtype=float)
    return 2.0 * np.pi * n_values * BOLTZMANN * config.temperature_k / HBAR


def semi_infinite_quadrature(order: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Map Gauss-Legendre nodes on ``[0, 1]`` to ``[0, infinity)``.

    The helper remains public for compatibility. The finite-temperature
    calculator itself uses adaptive :mod:`scipy.integrate` quadrature.
    """
    validated_order = _positive_integer_or_none("order", order)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Quadrature scale must be finite and positive.")
    nodes, weights = np.polynomial.legendre.leggauss(validated_order)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    values = scale * unit_nodes / (1.0 - unit_nodes)
    jacobian = scale / (1.0 - unit_nodes) ** 2
    return values, unit_weights * jacobian


def _as_finite_scalar(value, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"{name} must return a scalar for scalar input.")
    scalar = float(array.item())
    if not np.isfinite(scalar):
        raise FloatingPointError(f"{name} returned a non-finite value.")
    return scalar


def _round_trip_factors(
    K: float,
    k_parallel: float,
    distance_m: float,
    left_plane: LayeredPlane,
    right_plane: LayeredPlane,
    medium_permittivity: PermittivityFunction,
    polarization: str,
    *,
    _kappa_medium: float | None = None,
) -> tuple[float, float, float, float]:
    """Return q, kappa, 1-q, and log(1-q) with stable near-unity math."""
    xi = K * C
    if _kappa_medium is None:
        eps_medium = _as_finite_scalar(medium_permittivity(xi), "medium_permittivity")
        kappa_medium = _as_finite_scalar(
            longitudinal_wavevector(eps_medium, xi, k_parallel),
            "longitudinal_wavevector",
        )
    else:
        kappa_medium = _kappa_medium
    r_left = _as_finite_scalar(
        left_plane.reflection(medium_permittivity, xi, k_parallel, polarization),
        "left reflection",
    )
    r_right = _as_finite_scalar(
        right_plane.reflection(medium_permittivity, xi, k_parallel, polarization),
        "right reflection",
    )
    reflection_product = r_left * r_right
    if not np.isfinite(reflection_product):
        raise FloatingPointError("Reflection product is non-finite.")

    attenuation = 2.0 * kappa_medium * distance_m
    if reflection_product > 0.0:
        log_q = float(np.log(reflection_product) - attenuation)
        if not np.isfinite(log_q):
            raise FloatingPointError("Round-trip logarithm is non-finite.")
        if log_q >= 0.0:
            raise FloatingPointError(
                "Round-trip factor must be smaller than one for a real Lifshitz integrand."
            )
        q_value = float(np.exp(log_q))
        denominator = float(-np.expm1(log_q))
        log_denominator = float(np.log(denominator))
    else:
        q_value = float(reflection_product * np.exp(-attenuation))
        denominator = 1.0 - q_value
        log_denominator = float(np.log1p(-q_value))

    if (
        not np.isfinite(q_value)
        or not np.isfinite(denominator)
        or not np.isfinite(log_denominator)
        or denominator <= 0.0
    ):
        raise FloatingPointError("Round-trip factors produced a non-finite integrand.")
    return q_value, kappa_medium, denominator, log_denominator


def _k_integrand(
    k_parallel: float,
    K: float,
    distance_m: float,
    left_plane: LayeredPlane,
    right_plane: LayeredPlane,
    medium_permittivity: PermittivityFunction,
    observable: str,
    *,
    _epsilon_medium: float | None = None,
) -> np.ndarray:
    if observable not in _VALID_OBSERVABLES:
        raise ValueError("observable must be energy, pressure, or pressuregradient.")
    if not np.isfinite(k_parallel) or k_parallel < 0.0:
        raise ValueError("k_parallel must be finite and non-negative.")
    if not np.isfinite(K) or K < 0.0:
        raise ValueError("K must be finite and non-negative.")

    # All three limiting integrands vanish at the single k_parallel = 0 point.
    # Returning the limit avoids 0 * log(0) and 0/0 for perfect reflectors.
    if k_parallel == 0.0:
        return np.zeros(2, dtype=float)

    # Both polarizations propagate through the same intervening medium.
    # The adaptive integral supplies epsilon(xi) prepared once per frequency.
    eps_medium = (
        _as_finite_scalar(medium_permittivity(K * C), "medium_permittivity")
        if _epsilon_medium is None
        else _epsilon_medium
    )
    kappa_medium = _as_finite_scalar(
        longitudinal_wavevector(eps_medium, K * C, k_parallel),
        "longitudinal_wavevector",
    )
    values = []
    for polarization in ("TE", "TM"):
        q_value, kappa_medium, denominator, log_denominator = _round_trip_factors(
            K,
            k_parallel,
            distance_m,
            left_plane,
            right_plane,
            medium_permittivity,
            polarization,
            _kappa_medium=kappa_medium,
        )
        if observable == "energy":
            value = k_parallel * log_denominator / (2.0 * np.pi)
        elif observable == "pressure":
            value = -k_parallel * kappa_medium * q_value / (np.pi * denominator)
        else:
            value = (
                2.0
                * k_parallel
                * kappa_medium**2
                * q_value
                / (np.pi * denominator**2)
            )
        if not np.isfinite(value):
            raise FloatingPointError(
                f"Non-finite {observable} integrand for {polarization} polarization."
            )
        values.append(value)
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class _PreparedLayeredPlane:
    """One frequency's optical inputs, without repeated material evaluation."""

    epsilons: tuple[np.ndarray, ...]
    thicknesses_m: tuple[float, ...]

    def reflection(self, medium_permittivity, xi, k_parallel, polarization):
        # Retain the existing multilayer recurrence and zero-frequency limits.
        return layered_reflection(
            self.epsilons, self.thicknesses_m, xi, k_parallel, polarization
        )


def _prepare_frequency(
    xi: float,
    left_plane: LayeredPlane,
    right_plane: LayeredPlane,
    medium_permittivity: PermittivityFunction,
):
    """Evaluate each distinct built-in plane material once at this frequency.

    The cache is local to one integral, so edited material data is respected on
    later calls and continuous-frequency calculations cannot grow a global
    cache. Callable objects need not be hashable. Custom planes and subclasses
    keep their own reflection method and the original medium callback.
    """
    epsilon_values: dict[int, np.ndarray] = {}

    def evaluate(material: PermittivityFunction) -> np.ndarray:
        key = id(material)
        if key not in epsilon_values:
            epsilon_values[key] = np.array(material(xi), dtype=float, copy=True)
        return epsilon_values[key]

    medium_value = evaluate(medium_permittivity)
    epsilon_medium = _as_finite_scalar(medium_value, "medium_permittivity")

    def prepare(plane):
        # An isinstance check would bypass overridden subclass reflections.
        if type(plane) is not LayeredPlane:
            return plane
        return _PreparedLayeredPlane(
            (medium_value,) + tuple(evaluate(material) for material in plane.materials),
            plane.thicknesses_m,
        )

    return epsilon_medium, prepare(left_plane), prepare(right_plane)


def _integrate_vector_on_positive_axis(
    function: Callable[[float], np.ndarray],
    *,
    epsrel: float,
    epsabs: float,
    description: str,
) -> np.ndarray:
    """Integrate a TE/TM vector on ``[0, infinity)`` and verify convergence."""
    _validate_tolerances(epsrel, epsabs)
    result, error, info = integrate.quad_vec(
        function,
        0.0,
        np.inf,
        epsrel=epsrel,
        epsabs=epsabs,
        norm="max",
        full_output=True,
    )
    result_array = np.asarray(result, dtype=float)
    error_value = float(error)
    if result_array.shape != (2,):
        raise RuntimeError(
            f"{description} returned shape {result_array.shape}; expected TE/TM shape (2,)."
        )
    if not np.all(np.isfinite(result_array)) or not np.isfinite(error_value):
        raise FloatingPointError(f"{description} returned a non-finite result or error estimate.")
    if not bool(getattr(info, "success", False)):
        status = getattr(info, "status", "unknown")
        message = getattr(info, "message", "no diagnostic message")
        raise RuntimeError(
            f"{description} did not converge (status={status}, error={error_value:.3e}): "
            f"{message}"
        )
    return result_array


def frequency_function(
    observable: str,
    distance_m: float,
    left_plane: LayeredPlane,
    right_plane: LayeredPlane,
    medium_permittivity: PermittivityFunction,
    epsrel: float = 1.0e-8,
    epsabs: float = 0.0,
) -> Callable[[float], np.ndarray]:
    """Return the K-dependent TE/TM contribution used in the Matsubara sum."""
    if observable not in _VALID_OBSERVABLES:
        raise ValueError("observable must be energy, pressure, or pressuregradient.")
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("distance_m must be finite and positive.")
    _validate_tolerances(epsrel, epsabs)
    integration_epsabs = _effective_epsabs(
        epsabs,
        distance_m,
        _OBSERVABLE_DISTANCE_POWER[observable],
    )

    def contribution(K: float) -> np.ndarray:
        if not np.isfinite(K) or K < 0.0:
            raise ValueError("K must be finite and non-negative.")
        epsilon_medium, prepared_left, prepared_right = _prepare_frequency(
            K * C, left_plane, right_plane, medium_permittivity
        )

        def scaled_integrand(scaled_k: float) -> np.ndarray:
            return _k_integrand(
                scaled_k / distance_m,
                K,
                distance_m,
                prepared_left,
                prepared_right,
                medium_permittivity,
                observable,
                _epsilon_medium=epsilon_medium,
            ) / distance_m

        return _integrate_vector_on_positive_axis(
            scaled_integrand,
            epsrel=epsrel,
            epsabs=integration_epsabs,
            description=f"radial wave-vector integral at K={K:.6e}",
        )

    return contribution


def calculate_per_area(
    observable: str,
    distance_m: float,
    left_plane: LayeredPlane,
    right_plane: LayeredPlane,
    medium_permittivity: PermittivityFunction,
    config: FiniteTemperatureConfig,
    ht_limit: bool = False,
) -> float:
    """Calculate plane-plane energy, pressure, or pressure gradient."""
    if observable not in _VALID_OBSERVABLES:
        raise ValueError("observable must be energy, pressure, or pressuregradient.")
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("distance_m must be finite and positive.")
    func = frequency_function(
        observable,
        distance_m,
        left_plane,
        right_plane,
        medium_permittivity,
        epsrel=config.epsrel,
        epsabs=config.epsabs,
    )

    if config.temperature_k == 0.0:
        frequency_epsabs = _effective_epsabs(
            config.epsabs,
            distance_m,
            _OBSERVABLE_DISTANCE_POWER[observable] + 1,
        )
        zero_temperature_te_tm = _integrate_vector_on_positive_axis(
            lambda scaled_K: func(scaled_K / distance_m) / distance_m,
            epsrel=config.epsrel,
            epsabs=frequency_epsabs,
            description="zero-temperature frequency integral",
        )
        result = HBAR * C * np.sum(zero_temperature_te_tm) / (2.0 * np.pi)
        if not np.isfinite(result):
            raise FloatingPointError("Zero-temperature result is non-finite.")
        return float(result)

    n0_te_tm = 0.5 * config.temperature_k * BOLTZMANN * func(0.0)
    if not np.all(np.isfinite(n0_te_tm)):
        raise FloatingPointError("The n=0 Matsubara contribution is non-finite.")
    if ht_limit:
        return float(np.sum(n0_te_tm))

    summation = psd_sum if config.fs == "psd" else msd_sum
    n1_te_tm = np.asarray(
        summation(
            config.temperature_k,
            distance_m,
            func,
            config.epsrel,
            order=config.frequency_order,
        ),
        dtype=float,
    )
    if n1_te_tm.shape != (2,) or not np.all(np.isfinite(n1_te_tm)):
        raise FloatingPointError("The finite-frequency contribution is non-finite.")
    result = np.sum(n0_te_tm + n1_te_tm)
    if not np.isfinite(result):
        raise FloatingPointError("Finite-temperature result is non-finite.")
    return float(result)


def free_energy_per_area(
    distance_m: float,
    left_plane: LayeredPlane,
    right_plane: LayeredPlane,
    medium_permittivity: PermittivityFunction,
    config: FiniteTemperatureConfig,
) -> float:
    return calculate_per_area(
        "energy",
        distance_m,
        left_plane,
        right_plane,
        medium_permittivity,
        config,
    )


def pressure_per_area(
    distance_m: float,
    left_plane: LayeredPlane,
    right_plane: LayeredPlane,
    medium_permittivity: PermittivityFunction,
    config: FiniteTemperatureConfig,
) -> float:
    return calculate_per_area(
        "pressure",
        distance_m,
        left_plane,
        right_plane,
        medium_permittivity,
        config,
    )


def pressure_terms_by_matsubara(
    distance_m: float,
    left_plane: LayeredPlane,
    right_plane: LayeredPlane,
    medium_permittivity: PermittivityFunction,
    config: FiniteTemperatureConfig,
) -> list[dict[str, float | int]]:
    """Return n=0 plus the configured number of direct MSD pressure terms.

    A Matsubara decomposition is discrete only at positive temperature.  At
    zero temperature the pressure is instead a continuous-frequency integral,
    so returning zero-valued rows would be physically misleading.
    """
    if config.temperature_k == 0.0:
        raise ValueError(
            "pressure_terms_by_matsubara requires positive temperature; "
            "at zero temperature use pressure_per_area for the continuous-frequency integral."
        )

    order = config.frequency_order if config.frequency_order is not None else 40
    n_values = np.arange(order + 1)
    xi_values = 2.0 * np.pi * n_values * BOLTZMANN * config.temperature_k / HBAR
    func = frequency_function(
        "pressure",
        distance_m,
        left_plane,
        right_plane,
        medium_permittivity,
        epsrel=config.epsrel,
        epsabs=config.epsabs,
    )

    rows: list[dict[str, float | int]] = []
    for n, xi in enumerate(xi_values):
        weight = 0.5 if n == 0 else 1.0
        te_tm = weight * BOLTZMANN * config.temperature_k * func(float(xi / C))
        if not np.all(np.isfinite(te_tm)):
            raise FloatingPointError(f"Matsubara pressure term n={n} is non-finite.")
        rows.append(
            {
                "n": n,
                "xi": float(xi),
                "P_TE": float(te_tm[0]),
                "P_TM": float(te_tm[1]),
                "P_total": float(np.sum(te_tm)),
            }
        )
    return rows


__all__ = [
    "FiniteTemperatureConfig",
    "calculate_per_area",
    "free_energy_per_area",
    "frequency_function",
    "matsubara_frequencies",
    "pressure_per_area",
    "pressure_terms_by_matsubara",
    "semi_infinite_quadrature",
]
