from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.constants import c

from .constants import EV_TO_RAD_S
from .reference_materials import (
    clear_reference_material_cache,
    cylinder_reference_provenance,
    ethanol_cylinder_reference_permittivity,
    gold_cylinder_reference_permittivity,
    teflon_cylinder_reference_permittivity,
)

_MATERIAL_DATA_DIR = Path(__file__).resolve().parent / "material_data"
# AF2400 data-driven/diagnostic fits are disabled for now. The only active
# AF2400 model is the Yang/French/Tokarsky 2008 Table 2 Sellmeier continuation
# in teflon_af2400_permittivity().
# _AF2400_LIFSHITZ_FIT_PATH = _MATERIAL_DATA_DIR / "teflon_af2400_lifshitz_fit.csv"
# _AF2400_YANG_KK_PATH = _MATERIAL_DATA_DIR / "teflon_af2400_yang_kk_epsilon.csv"
# _AF2400_DAMPED_FIT_PATH = _MATERIAL_DATA_DIR / "teflon_af2400_damped_fit.csv"

# Au Drude free-electron parameters.
# The plasma energy hbar*omega_p = 8.45 eV follows R. L. Olmon et al.,
# Phys. Rev. B 86, 235147 (2012), DOI: 10.1103/PhysRevB.86.235147.
_GOLD_PLASMA_EV = 8.45
_GOLD_GAMMA_EV = 0.047

# Benjamin's califorcia repository uses this simple Au Drude model.
_CALIFORCIA_GOLD_PLASMA_EV = 9.0
_CALIFORCIA_GOLD_GAMMA_EV = 0.035

# Au interband Lorentz oscillator parameters, based on Palik 1985 Table 13.1.
_AU_OSCILLATOR_FREQ_EV = np.array([3.05, 4.15, 5.4, 8.5, 13.5, 21.5], dtype=float)
_AU_OSCILLATOR_GAMMA_EV = np.array([0.75, 1.85, 1.0, 7.0, 6.0, 9.0], dtype=float)
_AU_OSCILLATOR_AMPLITUDE_EV2 = np.array([7.091, 41.46, 2.700, 154.7, 44.55, 309.6], dtype=float)

_AU_FREQ_RADS = _AU_OSCILLATOR_FREQ_EV * EV_TO_RAD_S
_AU_GAMMA_RADS = _AU_OSCILLATOR_GAMMA_EV * EV_TO_RAD_S
_AU_AMPLITUDE = _AU_OSCILLATOR_AMPLITUDE_EV2 * (EV_TO_RAD_S ** 2)

# Ethanol Lorentz oscillator parameters.
# Source used in the Zhao-style Lifshitz calculation:
# P. J. van Zwol and G. Palasantzas, Phys. Rev. A 81, 062502 (2010),
# DOI: 10.1103/PhysRevA.81.062502.
_ETHANOL_C = np.array([9.57e-1, 1.62e0, 1.40e-1, 1.26e-1, 4.16e-1, 2.44e-1, 7.10e-2], dtype=float)
_ETHANOL_OMEGA_EV = np.array([1.62e-3, 4.32e-3, 1.12e-1, 6.87e0, 1.52e1, 1.56e1, 4.38e1], dtype=float)
_ETHANOL_STATIC_EPSILON = 24.3

# PTFE parameters based on the eight-oscillator model by Zwol and Palasantzas.
_PTFE_ZWOL_C = np.array([
    9.30e-3, 1.83e-2, 1.39e-1, 1.12e-1,
    1.95e-1, 4.38e-1, 1.06e-1, 3.86e-2,
], dtype=float)
_PTFE_ZWOL_OMEGA_EV = np.array([
    3.00e-4, 7.60e-3, 5.57e-2, 1.26e-1,
    6.71e0, 1.86e1, 4.21e1, 7.76e1,
], dtype=float)

# Teflon AF two-pole Sellmeier parameters from Table 2 of
# M. K. Yang, R. H. French, and E. W. Tokarsky,
# J. Micro/Nanolith. MEMS MOEMS 7, 033010 (2008),
# DOI: 10.1117/1.2965541. The paper grades are AF1300, AF1601,
# and AF2400, and the C coefficients are in um^2.
# Eq. (4) is n^2(lambda) = A + B1 lambda^2/(lambda^2 - C1)
# + B2 lambda^2/(lambda^2 - C2). The imaginary-frequency function
# below applies the analytic continuation lambda^2 -> -(2 pi c / xi)^2.
# This remains an optical-band Sellmeier model, not a full KK reconstruction.
_AF1300_A = 1.517
_AF1300_B1 = 0.184
_AF1300_C1 = 0.016
_AF1300_B2 = 1.0
_AF1300_C2 = 104.66

_AF1601_A = 1.461
_AF1601_B1 = 0.226
_AF1601_C1 = 0.014
_AF1601_B2 = 1.0
_AF1601_C2 = 119.03

_AF2400_A = 1.456
_AF2400_B1 = 0.181
_AF2400_C1 = 0.014
_AF2400_B2 = 1.0
_AF2400_C2 = 159.28

# Modified oscillator fits from Gudarzi et al., Sci. Adv. 7, eabg2272 (2021),
# Supplementary Information Table S1. These models are kept separate from the
# Yang/French/Tokarsky AF2400 Sellmeier continuation above, because they are
# full imaginary-frequency epsilon(i xi) fits intended for Lifshitz calculations.
_GUDARZI2021_TEFLON_C = np.array([0.716, 0.3], dtype=float)
_GUDARZI2021_TEFLON_OMEGA_EV = np.array([16.4, 0.082], dtype=float)
_GUDARZI2021_TEFLON_ALPHA = np.array([1.79, 1.82], dtype=float)

_GUDARZI2021_ETHANOL_C = np.array([19.85, 2.9, 0.827], dtype=float)
_GUDARZI2021_ETHANOL_OMEGA_EV = np.array([4.1e-6, 4.4e-4, 15.3], dtype=float)
_GUDARZI2021_ETHANOL_ALPHA = np.array([1.002, 0.67, 1.84], dtype=float)
_GUDARZI2021_ETHANOL_OPTICAL_STATIC_EPSILON = 1.85

_GUDARZI2021_GOLD_UV_C = 6.5
_GUDARZI2021_GOLD_UV_OMEGA_EV = 5.9
_GUDARZI2021_GOLD_UV_ALPHA = 1.42
_GUDARZI2021_GOLD_PLASMA_EV = 9.1
_GUDARZI2021_GOLD_GAMMA_EV = 0.06

# Disabled AF2400 alternative coefficients, kept here as comments only.
# Use teflon_af2400_permittivity() for active calculations.
#
# Diagnostic constrained AF2400 oscillator model:
# _AF2400_CONSTRAINED_C = np.array([0.26, 0.45, 0.19], dtype=float)
# _AF2400_CONSTRAINED_OMEGA_EV = np.array([0.10, 7.5, 20.0], dtype=float)
#
# Drummond-Georgaklis-Chan 1996 two-oscillator model:
# _AF2400_DRUMMOND1996_STATIC_EPSILON = 1.904
# _AF2400_DRUMMOND1996_C = np.array([0.239, 0.665], dtype=float)
# _AF2400_DRUMMOND1996_OMEGA_EV = np.array([2.356e14, 2.522e16], dtype=float) / EV_TO_RAD_S


def _to_float_array(xi: float | np.ndarray) -> np.ndarray:
    """Convert the frequency input to a floating-point array."""
    return np.asarray(xi, dtype=float)


@lru_cache(maxsize=None)
def _load_lifshitz_oscillator_fit(path_text: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Lifshitz oscillator fit file: {path}. "
            "Digitize Zhao Supplementary Fig. S1 and run fit_teflon_lifshitz_model.py first."
        )

    with path.open("r", newline="", encoding="utf-8") as handle:
        filtered_lines = [
            line.lstrip("\ufeff")
            for line in handle
            if line.strip() and not line.lstrip("\ufeff").lstrip().startswith("#")
        ]
    if not filtered_lines:
        raise ValueError(
            f"No oscillator rows found in {path}. "
            "Run fit_teflon_lifshitz_model.py after adding digitized data."
        )

    import csv

    reader = csv.DictReader(filtered_lines)
    if reader.fieldnames is not None:
        reader.fieldnames = [field.lstrip("\ufeff").strip() for field in reader.fieldnames]
    if reader.fieldnames is None or "omega_eV" not in reader.fieldnames or "strength" not in reader.fieldnames:
        raise ValueError(f"{path} must contain omega_eV and strength columns.")

    rows: list[tuple[float, float]] = []
    for row in reader:
        try:
            omega_value = float(row["omega_eV"])
            strength_value = float(row["strength"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(omega_value) and np.isfinite(strength_value) and omega_value > 0.0 and strength_value >= 0.0:
            rows.append((omega_value, strength_value))

    if not rows:
        raise ValueError(f"No valid nonnegative oscillator rows found in {path}.")

    data = np.array(rows, dtype=float)
    return data[:, 0], data[:, 1]


def _oscillator_fit_permittivity(
        xi: float | np.ndarray,
        omega_eV: np.ndarray,
        strengths: np.ndarray,
) -> np.ndarray:
    xi_array = _to_float_array(xi)
    original_shape = xi_array.shape
    xi_ev = np.atleast_1d(xi_array / EV_TO_RAD_S).reshape(-1, 1)
    values = 1.0 + np.sum(strengths / (1.0 + (xi_ev / omega_eV) ** 2), axis=1)
    return values.reshape(original_shape)


@lru_cache(maxsize=None)
def _load_damped_lorentz_fit(path_text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing damped Lorentz fit file: {path}. Run fit_af2400_damped_oscillator.py first."
        )

    import csv

    rows: list[tuple[float, float, float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(
            line for line in handle if line.strip() and not line.lstrip("﻿").lstrip().startswith("#")
        )
        if reader.fieldnames is not None:
            reader.fieldnames = [field.lstrip("﻿").strip() for field in reader.fieldnames]
        required = {"omega_eV", "strength", "gamma_eV"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain omega_eV, strength, and gamma_eV columns.")
        for row in reader:
            try:
                omega = float(row["omega_eV"])
                strength = float(row["strength"])
                gamma = float(row["gamma_eV"])
            except (TypeError, ValueError):
                continue
            if (
                np.isfinite(omega)
                and np.isfinite(strength)
                and np.isfinite(gamma)
                and omega > 0.0
                and strength >= 0.0
                and gamma >= 0.0
            ):
                rows.append((omega, strength, gamma))

    if not rows:
        raise ValueError(f"No valid damped Lorentz oscillator rows found in {path}.")

    data = np.array(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2]


def _damped_lorentz_imaginary_permittivity(
        xi: float | np.ndarray,
        omega_eV: np.ndarray,
        strengths: np.ndarray,
        gamma_eV: np.ndarray,
) -> np.ndarray:
    xi_array = _to_float_array(xi)
    original_shape = xi_array.shape
    xi_ev = np.atleast_1d(xi_array / EV_TO_RAD_S).reshape(-1, 1)
    omega = omega_eV.reshape(1, -1)
    strengths_2d = strengths.reshape(1, -1)
    gamma = gamma_eV.reshape(1, -1)
    values = 1.0 + np.sum(
        strengths_2d * omega ** 2 / (omega ** 2 + xi_ev ** 2 + gamma * xi_ev),
        axis=1,
    )
    return values.reshape(original_shape)


@lru_cache(maxsize=None)
def _load_tabulated_epsilon(path_text: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Missing tabulated epsilon file: {path}.")

    import csv

    rows: list[tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(
            line for line in handle if line.strip() and not line.lstrip("﻿").lstrip().startswith("#")
        )
        if reader.fieldnames is not None:
            reader.fieldnames = [field.lstrip("﻿").strip() for field in reader.fieldnames]
        if reader.fieldnames is None or "xi_eV" not in reader.fieldnames or "epsilon" not in reader.fieldnames:
            raise ValueError(f"{path} must contain xi_eV and epsilon columns.")
        for row in reader:
            try:
                xi_ev = float(row["xi_eV"])
                epsilon = float(row["epsilon"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(xi_ev) and np.isfinite(epsilon) and xi_ev >= 0.0 and epsilon >= 1.0:
                rows.append((xi_ev, epsilon))

    if not rows:
        raise ValueError(f"No valid tabulated epsilon rows found in {path}.")

    data = np.array(sorted(rows), dtype=float)
    return data[:, 0], data[:, 1]


def _tabulated_epsilon_permittivity(
        xi: float | np.ndarray,
        xi_eV: np.ndarray,
        epsilon_values: np.ndarray,
) -> np.ndarray:
    xi_array = _to_float_array(xi)
    original_shape = xi_array.shape
    xi_ev = np.atleast_1d(xi_array / EV_TO_RAD_S)

    if xi_eV[0] == 0.0:
        interp_x = xi_eV[1:]
        interp_y = epsilon_values[1:]
        zero_value = epsilon_values[0]
    else:
        interp_x = xi_eV
        interp_y = epsilon_values
        zero_value = epsilon_values[0]

    safe_x = np.clip(xi_ev, interp_x[0], interp_x[-1])
    # Log-x interpolation is smoother for imaginary-frequency material tables.
    values = np.interp(np.log(safe_x), np.log(interp_x), interp_y)
    values = np.where(xi_ev <= 0.0, zero_value, values)
    return values.reshape(original_shape)


def _modified_oscillator_permittivity(
        xi: float | np.ndarray,
        strengths: np.ndarray,
        resonance_ev: np.ndarray,
        exponents: np.ndarray,
) -> np.ndarray:
    """Compute epsilon(i xi) = 1 + sum C_j / (1 + (xi / omega_j)^alpha_j)."""
    xi_array = _to_float_array(xi)
    original_shape = xi_array.shape
    xi_ev = np.atleast_1d(np.abs(xi_array) / EV_TO_RAD_S).reshape(-1, 1)
    values = 1.0 + np.sum(
        strengths.reshape(1, -1)
        / (1.0 + (xi_ev / resonance_ev.reshape(1, -1)) ** exponents.reshape(1, -1)),
        axis=1,
    )
    return values.reshape(original_shape)


def _lorentz_sum_permittivity(
        xi: float | np.ndarray,
        strengths: np.ndarray,
        resonance_ev: np.ndarray,
        static_epsilon: float | None = None,
) -> np.ndarray:
    """Compute epsilon(i xi) = 1 + sum_j C_j / (1 + (xi / omega_j)^2)."""
    xi_array = _to_float_array(xi)
    resonance = resonance_ev * EV_TO_RAD_S
    result = np.zeros_like(xi_array, dtype=float)

    mask_zero = np.abs(xi_array) < 1e-16
    mask_nonzero = ~mask_zero

    if np.any(mask_zero):
        result[mask_zero] = static_epsilon if static_epsilon is not None else 1.0 + np.sum(strengths)

    if np.any(mask_nonzero):
        xi_nonzero = xi_array[mask_nonzero]
        ratio_squared = (xi_nonzero[:, np.newaxis] / resonance) ** 2
        result[mask_nonzero] = 1.0 + np.sum(strengths / (1.0 + ratio_squared), axis=1)

    return result


def vacuum_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Return the vacuum permittivity on the imaginary-frequency axis."""
    xi_array = _to_float_array(xi)
    return np.ones_like(xi_array, dtype=float)


def gold_drude_permittivity(
        xi: float | np.ndarray,
        plasma_frequency_ev: float = _GOLD_PLASMA_EV,
        damping_ev: float = _GOLD_GAMMA_EV,
) -> np.ndarray:
    """Compute the gold permittivity using a simple Drude model."""
    xi_array = _to_float_array(xi)
    omega_p = plasma_frequency_ev * EV_TO_RAD_S
    gamma = damping_ev * EV_TO_RAD_S

    result = np.ones_like(xi_array, dtype=np.float64)
    near_zero = np.abs(xi_array) < 1e-16

    if np.any(~near_zero):
        xi_nonzero = xi_array[~near_zero]
        result[~near_zero] = 1.0 + omega_p ** 2 / (xi_nonzero * (xi_nonzero + gamma))

    if np.any(near_zero):
        # The zero-frequency limit diverges; use a large finite value to avoid overflow.
        result[near_zero] = 1.0 + omega_p ** 2 / (1e-32 * gamma)

    return result


def gold_califorcia_drude_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Au Drude epsilon(i xi) matching Benjamin's califorcia material model."""
    return gold_drude_permittivity(
        xi,
        plasma_frequency_ev=_CALIFORCIA_GOLD_PLASMA_EV,
        damping_ev=_CALIFORCIA_GOLD_GAMMA_EV,
    )


def gold_lorentz_drude_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Au epsilon(i xi), Drude free-electron term plus Lorentz oscillators.

    The Drude plasma energy hbar*omega_p = 8.45 eV is taken from
    R. L. Olmon et al., Phys. Rev. B 86, 235147 (2012),
    DOI: 10.1103/PhysRevB.86.235147. The Lorentz oscillator block keeps the
    existing Palik-based interband parameters.
    """
    xi_array = _to_float_array(xi)
    original_shape = xi_array.shape
    xi_flat = np.atleast_1d(xi_array).reshape(-1, 1)

    omega_p = _GOLD_PLASMA_EV * EV_TO_RAD_S
    gamma = _GOLD_GAMMA_EV * EV_TO_RAD_S

    # The Drude term diverges as xi -> 0. Use a tiny finite xi to keep the
    # numerical Fresnel calculation finite while preserving the metallic limit.
    xi_safe = np.where(np.abs(xi_flat) < 1.0e-16, 1.0e-16, xi_flat)
    drude_term = omega_p ** 2 / (xi_safe * (xi_safe + gamma))

    denominator = _AU_FREQ_RADS ** 2 + xi_flat ** 2 + _AU_GAMMA_RADS * xi_flat
    lorentz_term = np.sum(_AU_AMPLITUDE / denominator, axis=1, keepdims=True)

    result = 1.0 + drude_term + lorentz_term
    return result.reshape(original_shape)


def ethanol_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Compute the ethanol permittivity on the imaginary-frequency axis."""
    return _lorentz_sum_permittivity(
        xi,
        strengths=_ETHANOL_C,
        resonance_ev=_ETHANOL_OMEGA_EV,
        static_epsilon=_ETHANOL_STATIC_EPSILON,
    )


def ptfe_zwol_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Compute the PTFE permittivity using the Zwol-Palasantzas eight-oscillator model."""
    return _lorentz_sum_permittivity(
        xi,
        strengths=_PTFE_ZWOL_C,
        resonance_ev=_PTFE_ZWOL_OMEGA_EV,
    )


def ptfe_ld_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Return the permittivity of low-density PTFE."""
    return ptfe_zwol_permittivity(xi)


def ptfe_hd_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Return the permittivity of high-density PTFE."""
    return ptfe_zwol_permittivity(xi)


def teflon_gudarzi2021_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Teflon epsilon(i xi) from Gudarzi et al. 2021 Supplementary Table S1.

    The fit is a modified oscillator model:
        epsilon(i xi) = 1 + 0.716 / (1 + (xi_eV / 16.4)^1.79)
                          + 0.3 / (1 + (xi_eV / 0.082)^1.82)

    This is not the Yang AF2400 Sellmeier continuation; it is a full
    imaginary-frequency model used for Lifshitz calculations.
    """
    return _modified_oscillator_permittivity(
        xi,
        strengths=_GUDARZI2021_TEFLON_C,
        resonance_ev=_GUDARZI2021_TEFLON_OMEGA_EV,
        exponents=_GUDARZI2021_TEFLON_ALPHA,
    )


def ethanol_gudarzi2021_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Ethanol epsilon(i xi) from Gudarzi et al. 2021 Supplementary Table S1."""
    return _modified_oscillator_permittivity(
        xi,
        strengths=_GUDARZI2021_ETHANOL_C,
        resonance_ev=_GUDARZI2021_ETHANOL_OMEGA_EV,
        exponents=_GUDARZI2021_ETHANOL_ALPHA,
    )


def ethanol_gudarzi2021_optical_static_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Diagnostic ethanol model with only epsilon(0) replaced by optical n^2.

    All nonzero imaginary-frequency values use the Gudarzi 2021 fit. The exact
    zero-frequency point is set to 1.85 to test the sensitivity of the
    Matsubara n=0 TM term to ethanol's static orientational response.
    """
    xi_array = _to_float_array(xi)
    epsilon = ethanol_gudarzi2021_permittivity(xi_array)
    return np.where(
        np.abs(xi_array) < 1.0e-16,
        _GUDARZI2021_ETHANOL_OPTICAL_STATIC_EPSILON,
        epsilon,
    )


def gold_gudarzi2021_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Gold epsilon(i xi) from Gudarzi et al. 2021 Supplementary Table S1.

    The metal is represented by one modified UV oscillator plus a Drude term
    with omega_p = 9.1 eV and gamma = 0.06 eV.
    """
    xi_array = _to_float_array(xi)
    original_shape = xi_array.shape
    xi_ev = np.atleast_1d(np.abs(xi_array) / EV_TO_RAD_S).reshape(-1, 1)
    xi_safe = np.where(xi_ev < 1.0e-18, 1.0e-18, xi_ev)

    uv_term = _GUDARZI2021_GOLD_UV_C / (
        1.0 + (xi_safe / _GUDARZI2021_GOLD_UV_OMEGA_EV) ** _GUDARZI2021_GOLD_UV_ALPHA
    )
    drude_term = _GUDARZI2021_GOLD_PLASMA_EV ** 2 / (
        xi_safe * (xi_safe + _GUDARZI2021_GOLD_GAMMA_EV)
    )
    values = 1.0 + uv_term + drude_term
    return values.reshape(original_shape)


def _teflon_af_sellmeier_permittivity(
        xi: float | np.ndarray,
        a: float,
        b1: float,
        c1: float,
        b2: float,
        c2: float,
) -> np.ndarray:
    """Compute the Teflon AF permittivity on the imaginary-frequency axis using the Sellmeier form."""
    xi_array = _to_float_array(xi)
    xi_safe = np.where(np.abs(xi_array) < 1e-16, 1e-16, xi_array)

    wavelength_um = 2.0 * np.pi * c / xi_safe * 1e6
    denominator1 = np.where(np.abs(wavelength_um ** 2 + c1) < 1e-16, 1e-16, wavelength_um ** 2 + c1)
    denominator2 = np.where(np.abs(wavelength_um ** 2 + c2) < 1e-16, 1e-16, wavelength_um ** 2 + c2)

    return a + b1 * wavelength_um ** 2 / denominator1 + b2 * wavelength_um ** 2 / denominator2


def teflon_af1300_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Compute the Teflon AF1300 permittivity."""
    return _teflon_af_sellmeier_permittivity(
        xi=xi,
        a=_AF1300_A,
        b1=_AF1300_B1,
        c1=_AF1300_C1,
        b2=_AF1300_B2,
        c2=_AF1300_C2,
    )


def teflon_af1601_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Teflon AF1601 epsilon(i xi) using Yang et al. Table 2 Eq. (4)."""
    return _teflon_af_sellmeier_permittivity(
        xi=xi,
        a=_AF1601_A,
        b1=_AF1601_B1,
        c1=_AF1601_C1,
        b2=_AF1601_B2,
        c2=_AF1601_C2,
    )


def teflon_af1600_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Backward-compatible alias for Teflon AF1601.

    Yang et al. (2008) label this grade as AF1601. Older scripts in this
    project used the broader name AF1600, so this alias is kept to avoid
    breaking existing runs.
    """
    return teflon_af1601_permittivity(xi)


def teflon_af2400_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Teflon AF2400 epsilon(i xi) from Yang/French/Tokarsky 2008.

    Active AF2400 model:
        A=1.456, B1=0.181, C1=0.014, B2=1.0, C2=159.28

    These are the AF2400 Table 2 Sellmeier parameters in
    TeflonAF_Opt_Prop.pdf. Other AF2400 diagnostic/fitted dielectric models
    are intentionally disabled below.
    """
    return _teflon_af_sellmeier_permittivity(
        xi=xi,
        a=_AF2400_A,
        b1=_AF2400_B1,
        c1=_AF2400_C1,
        b2=_AF2400_B2,
        c2=_AF2400_C2,
    )


def teflon_af2400_drummond1996_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Disabled AF2400 alternative model.

    Only teflon_af2400_permittivity(), based on TeflonAF_Opt_Prop.pdf Table 2,
    is active for AF2400 calculations.
    """
    raise RuntimeError("AF2400 Drummond1996 model is disabled; use coating_model='af2400'.")


def teflon_af2400_constrained_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Disabled AF2400 alternative model."""
    raise RuntimeError("AF2400 constrained model is disabled; use coating_model='af2400'.")


def teflon_af2400_damped_fit_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Disabled AF2400 alternative model."""
    raise RuntimeError("AF2400 damped-fit model is disabled; use coating_model='af2400'.")


def teflon_af2400_yang_kk_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Disabled AF2400 alternative model."""
    raise RuntimeError("AF2400 Yang-KK model is disabled; use coating_model='af2400'.")


def clear_material_model_cache() -> None:
    """Clear cached fits and upstream material modules after source/data edits."""
    _load_lifshitz_oscillator_fit.cache_clear()
    _load_tabulated_epsilon.cache_clear()
    _load_damped_lorentz_fit.cache_clear()
    clear_reference_material_cache()


def teflon_af2400_lifshitz_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Disabled AF2400 alternative model."""
    raise RuntimeError("AF2400 Lifshitz-fit model is disabled; use coating_model='af2400'.")

__all__ = [
    "EV_TO_RAD_S",
    "clear_material_model_cache",
    "cylinder_reference_provenance",
    "ethanol_cylinder_reference_permittivity",
    "ethanol_gudarzi2021_optical_static_permittivity",
    "ethanol_gudarzi2021_permittivity",
    "ethanol_permittivity",
    "gold_califorcia_drude_permittivity",
    "gold_cylinder_reference_permittivity",
    "gold_drude_permittivity",
    "gold_gudarzi2021_permittivity",
    "gold_lorentz_drude_permittivity",
    "ptfe_zwol_permittivity",
    "ptfe_ld_permittivity",
    "ptfe_hd_permittivity",
    "teflon_af1300_permittivity",
    "teflon_af1600_permittivity",
    "teflon_af1601_permittivity",
    "teflon_af2400_permittivity",
    "teflon_gudarzi2021_permittivity",
    "teflon_cylinder_reference_permittivity",
    "vacuum_permittivity",
]


