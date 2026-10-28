from __future__ import annotations

import importlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeAlias, cast

import numpy as np

from .constants import ETHANOL_DENSITY, G_ACCEL, GOLD_DENSITY, NANOMETER
from .material_sets import COATING_MODELS, Permittivity, select_gold_and_medium

PlaneSphereSystemFactory: TypeAlias = Callable[..., Any]
CaspwnMaterialTriple: TypeAlias = tuple["CaspwnMaterial", "CaspwnMaterial", "CaspwnMaterial"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def caspwn_source() -> Path:
    """Locate the external caspwn checkout without vendoring it here."""
    if env_path := os.environ.get("CASPWN_ROOT"):
        source = Path(env_path).expanduser().resolve()
        if (source / "caspwn" / "__init__.py").is_file():
            return source
        raise RuntimeError(
            "CASPWN_ROOT does not contain caspwn/__init__.py: "
            f"{source}"
        )

    candidates = [
        project_root() / "external" / "caspwn-main",
        project_root().parent / "caspwn-main",
        Path.home() / "Desktop" / "Casimir Program" / "Main programe" / "external" / "caspwn-main",
        Path.home() / "Desktop" / "caspwn-main",
    ]
    for candidate in candidates:
        if (candidate / "caspwn" / "__init__.py").is_file():
            return candidate.resolve()
    raise RuntimeError(
        "Could not locate caspwn. Set CASPWN_ROOT to the caspwn-main directory."
    )


def ensure_caspwn_on_path() -> Path:
    source = caspwn_source()
    source_text = str(source)

    loaded_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "caspwn" or name.startswith("caspwn.")
    }
    selected_package = source / "caspwn"

    def belongs_to_selected_package(module: Any) -> bool:
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            return False
        return Path(module_file).resolve().is_relative_to(selected_package)

    if any(not belongs_to_selected_package(module) for module in loaded_modules.values()):
        # A previously imported installation otherwise wins through sys.modules,
        # even after the requested checkout is put first on sys.path. Purging
        # all submodules also prevents a mixed backend after a root switch.
        for module_name in loaded_modules:
            del sys.modules[module_name]

    # Remove duplicate/older positions before giving the selected checkout
    # unambiguous precedence.
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)
    importlib.invalidate_caches()

    module = importlib.import_module("caspwn")
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError("Imported caspwn has no source file; cannot verify its backend.")
    imported_source = Path(module_file).resolve().parent.parent
    if imported_source != source:
        raise RuntimeError(
            "Imported caspwn from an unexpected location. "
            f"Requested {source}, imported {imported_source}."
        )
    return source


def load_plane_sphere_system() -> PlaneSphereSystemFactory:
    ensure_caspwn_on_path()
    return cast(
        PlaneSphereSystemFactory,
        getattr(importlib.import_module("caspwn"), "plane_sphere_system"),
    )


@dataclass(frozen=True)
class CaspwnMaterial:
    """Material adapter accepted by caspwn and by this project's plane code."""

    name: str
    materialclass: str
    permittivity: Permittivity

    def __call__(self, xi: float | np.ndarray) -> np.ndarray:
        return self.permittivity(xi)

    def epsilon(self, xi: float) -> float:
        return float(np.asarray(self.permittivity(xi), dtype=float))


@dataclass(frozen=True)
class NumericalSettings:
    """Optional overrides for caspwn's plane-sphere numerical defaults.

    ``None`` delegates the corresponding setting to caspwn. In particular,
    caspwn chooses ``N``, ``M`` and ``lmax`` from ``R/L`` and chooses the PSD
    order from ``epsrel``. An explicit ``psd_order`` disables that automatic
    PSD-order selection, so ``epsrel`` no longer controls frequency truncation.
    """

    n_radial: int | None = None
    m_angular: int | None = None
    psd_order: int | None = None
    epsrel: float | None = None
    lmax_factor: float | None = None
    min_lmax: int | None = None
    cores: int = 1

    def __post_init__(self) -> None:
        if self.n_radial is not None and self.n_radial <= 0:
            raise ValueError("Sphere quadrature order N must be positive.")
        if self.m_angular is not None and self.m_angular <= 0:
            raise ValueError("Sphere quadrature order M must be positive.")
        if self.psd_order is not None and self.psd_order < 2:
            raise ValueError("caspwn requires psd_order >= 2.")
        if self.epsrel is not None and (
            not np.isfinite(self.epsrel) or self.epsrel <= 0.0
        ):
            raise ValueError("epsrel must be finite and positive.")
        if self.lmax_factor is not None and (
            not np.isfinite(self.lmax_factor) or self.lmax_factor <= 0.0
        ):
            raise ValueError("lmax_factor must be finite and positive.")
        if self.min_lmax is not None and self.min_lmax <= 0:
            raise ValueError("min_lmax must be positive.")
        if self.cores <= 0:
            raise ValueError("cores must be positive.")


SCENARIOS = {
    0: "no_gravity",
    1: "positive_gravity",
    2: "negative_gravity",
}


def make_materials(material_set: str, coating_model: str) -> CaspwnMaterialTriple:
    if coating_model not in COATING_MODELS:
        allowed = ", ".join(sorted(COATING_MODELS))
        raise ValueError(f"Unsupported coating_model '{coating_model}'. Allowed: {allowed}")
    gold_eps, ethanol_eps = select_gold_and_medium(material_set)
    coating_eps = COATING_MODELS[coating_model]
    return (
        CaspwnMaterial(f"gold-{material_set}", "drude", gold_eps),
        CaspwnMaterial(f"ethanol-{material_set}", "dielectric", ethanol_eps),
        CaspwnMaterial(coating_model, "dielectric", coating_eps),
    )


def lmax_for(radius_m: float, distance_m: float, settings: NumericalSettings) -> int | None:
    if settings.lmax_factor is None and settings.min_lmax is None:
        return None
    rho = max(radius_m / distance_m, 50.0)
    # 12*rho is caspwn's own plane-sphere default. It is used only when a
    # caller requests a minimum lmax without specifying a custom factor.
    factor = 12.0 if settings.lmax_factor is None else settings.lmax_factor
    minimum = 0 if settings.min_lmax is None else settings.min_lmax
    return max(minimum, int(factor * rho))


def caspwn_observable(
    observable: str,
    radius_m: float,
    distance_m: float,
    teflon_thickness_m: float,
    temperature_k: float,
    materials: CaspwnMaterialTriple,
    settings: NumericalSettings,
) -> float:
    """Evaluate one native plane-sphere observable without changing the backend."""
    if observable not in {"energy", "force", "forcegradient"}:
        raise ValueError("observable must be 'energy', 'force', or 'forcegradient'.")
    if not np.isfinite(teflon_thickness_m) or teflon_thickness_m < 0.0:
        raise ValueError("teflon_thickness_m must be finite and non-negative.")
    gold, ethanol, coating = materials
    plane_surface = gold if teflon_thickness_m == 0.0 else coating
    system = load_plane_sphere_system()(
        T=temperature_k,
        L=distance_m,
        R=radius_m,
        mat_sphere=gold,
        mat_plane=plane_surface,
        mat_medium=ethanol,
    )
    if teflon_thickness_m > 0.0:
        system.add_plane_layers([gold], [teflon_thickness_m])
    calculate_kwargs: dict[str, int | float] = {"cores": settings.cores}
    optional_overrides = {
        "N": settings.n_radial,
        "M": settings.m_angular,
        "lmax": lmax_for(radius_m, distance_m, settings),
        "O": settings.psd_order,
        "epsrel": settings.epsrel,
    }
    calculate_kwargs.update(
        (name, value) for name, value in optional_overrides.items() if value is not None
    )
    return float(system.calculate(observable, **calculate_kwargs))


def caspwn_energy_joule(
    radius_m: float,
    distance_m: float,
    teflon_thickness_m: float,
    temperature_k: float,
    materials: CaspwnMaterialTriple,
    settings: NumericalSettings,
) -> float:
    """Return the backend's total sphere-plane interaction energy in joules."""
    return caspwn_observable(
        "energy",
        radius_m,
        distance_m,
        teflon_thickness_m,
        temperature_k,
        materials,
        settings,
    )


def caspwn_force_newton(
    radius_m: float,
    distance_m: float,
    teflon_thickness_m: float,
    temperature_k: float,
    materials: CaspwnMaterialTriple,
    settings: NumericalSettings,
) -> float:
    return caspwn_observable(
        "force",
        radius_m,
        distance_m,
        teflon_thickness_m,
        temperature_k,
        materials,
        settings,
    )


def compute_force_curve(
    radius_nm: float,
    distances_nm: np.ndarray,
    teflon_thickness_m: float,
    temperature_k: float,
    materials: CaspwnMaterialTriple,
    settings: NumericalSettings,
) -> np.ndarray:
    radius_m = radius_nm * NANOMETER
    forces = [
        caspwn_force_newton(
            radius_m=radius_m,
            distance_m=float(distance_nm) * NANOMETER,
            teflon_thickness_m=teflon_thickness_m,
            temperature_k=temperature_k,
            materials=materials,
            settings=settings,
        )
        for distance_nm in distances_nm
    ]
    return np.array(forces, dtype=float)


def buoyant_gravity_force_newton(
    radius_m: float,
    scenario: int,
    sphere_density: float = GOLD_DENSITY,
    medium_density: float = ETHANOL_DENSITY,
) -> float:
    if scenario == 0:
        return 0.0
    if scenario not in {1, 2}:
        raise ValueError("scenario must be 0, 1, or 2")
    volume = 4.0 * math.pi * radius_m**3 / 3.0
    magnitude = (sphere_density - medium_density) * volume * G_ACCEL
    return -magnitude if scenario == 1 else magnitude


__all__ = [
    "CaspwnMaterial",
    "CaspwnMaterialTriple",
    "NumericalSettings",
    "SCENARIOS",
    "buoyant_gravity_force_newton",
    "caspwn_energy_joule",
    "caspwn_force_newton",
    "caspwn_observable",
    "caspwn_source",
    "compute_force_curve",
    "ensure_caspwn_on_path",
    "lmax_for",
    "load_plane_sphere_system",
    "make_materials",
    "project_root",
]
