"""Validated numerical adapter for the teacher cylinder-plane scattering code.

This module has no command-line, plotting, or campaign dependencies. The
upstream compute_cypl.system owns scattering, layered reflections, quadrature,
frequency summation and the corrected force-gradient trace calculation.
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

import numpy as np

from .material_sets import DEFAULT_MATERIAL_SET, DEFAULT_COATING_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MATERIAL_SET = DEFAULT_MATERIAL_SET
REQUIRED_COATING_MODEL = DEFAULT_COATING_MODEL
REQUIRED_ROLE_MATERIALS = {
    "cylinder_material": "gold", "medium_material": "ethanol",
    "coating_material": "teflon", "substrate_material": "gold",
}




def find_cylinder_repo(project_root: Path) -> Path:
    env_path = os.environ.get("CASIMIR_CYLINDER_ROOT")
    if env_path is not None:
        if not env_path.strip():
            raise RuntimeError("CASIMIR_CYLINDER_ROOT must not be empty.")
        source = Path(env_path).expanduser().resolve()
        if (source / "casimir_cylinder" / "compute_cypl.py").is_file():
            return source
        raise RuntimeError(
            "CASIMIR_CYLINDER_ROOT does not contain "
            f"casimir_cylinder/compute_cypl.py: {source}"
        )

    desktop = Path.home() / "Desktop"
    candidates = [
        project_root / "external" / "casimir_cylinder-main",
        project_root.parent / "external" / "casimir_cylinder-main",
        desktop / "casimir_cylinder-main",
        project_root.parent / "casimir_cylinder-main",
    ]
    for candidate in candidates:
        if (candidate / "casimir_cylinder" / "compute_cypl.py").is_file():
            return candidate.resolve()
    raise RuntimeError(
        "Could not locate casimir_cylinder-main. Set CASIMIR_CYLINDER_ROOT to the repository path."
    )


def put_path_first(path: Path) -> None:
    """Give one resolved checkout precedence without retaining duplicates."""
    resolved = path.resolve()

    def is_selected_path(entry: str) -> bool:
        try:
            return Path(entry).resolve() == resolved
        except (OSError, RuntimeError):
            return False

    sys.path[:] = [entry for entry in sys.path if not is_selected_path(entry)]
    sys.path.insert(0, str(resolved))


CYLINDER_REPO = find_cylinder_repo(PROJECT_ROOT)
_cylinder_system_factory = None
_normalize_plane_stack = None
fcqs_combined = None
fcqs_semiinfinite = None
_LOADED_BACKEND_REPO: Path | None = None


def ensure_cylinder_backend_source() -> bool:
    """Select one backend checkout and purge any previously loaded competitor."""
    selected_repo = CYLINDER_REPO.resolve()
    selected_package = selected_repo / "casimir_cylinder"
    loaded_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "casimir_cylinder" or name.startswith("casimir_cylinder.")
    }

    def belongs_to_selected_package(module: object) -> bool:
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            return False
        try:
            return Path(module_file).resolve().is_relative_to(selected_package)
        except (OSError, RuntimeError):
            return False

    source_mismatch = any(
        not belongs_to_selected_package(module) for module in loaded_modules.values()
    )
    if source_mismatch:
        # Removing every child avoids a mixed package when a different checkout
        # has already populated sys.modules.
        for module_name in loaded_modules:
            del sys.modules[module_name]

    put_path_first(selected_repo)
    importlib.invalidate_caches()
    package = importlib.import_module("casimir_cylinder")
    package_file = getattr(package, "__file__", None)
    if package_file is None:
        raise RuntimeError(
            "Imported casimir_cylinder has no source file; cannot verify its backend."
        )
    imported_repo = Path(package_file).resolve().parent.parent
    if imported_repo != selected_repo:
        raise RuntimeError(
            "Imported casimir_cylinder from an unexpected location. "
            f"Requested {selected_repo}, imported {imported_repo}."
        )
    return source_mismatch


def load_cylinder_backend() -> None:
    """Load the expensive scattering backend only when a scan starts."""
    global _cylinder_system_factory, _normalize_plane_stack
    global fcqs_combined, fcqs_semiinfinite
    global _LOADED_BACKEND_REPO

    source_mismatch = ensure_cylinder_backend_source()
    selected_repo = CYLINDER_REPO.resolve()
    if (
        _cylinder_system_factory is not None
        and _LOADED_BACKEND_REPO == selected_repo
        and not source_mismatch
    ):
        return

    compute = importlib.import_module("casimir_cylinder.compute_cypl")
    plane = importlib.import_module("casimir_cylinder.plane")
    quadratures = importlib.import_module("casimir_cylinder.quadratures")
    for module in (compute, plane, quadratures):
        module_file = getattr(module, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(
            selected_repo / "casimir_cylinder"
        ):
            raise RuntimeError(
                f"Imported {module.__name__} from outside the selected cylinder backend: "
                f"{module_file}"
            )
    if (
        not hasattr(compute, "system")
        or "plane_thicknesses" not in inspect.signature(compute.system).parameters
        or not callable(getattr(plane, "normalize_plane_stack", None))
    ):
        raise RuntimeError(
            "The selected casimir_cylinder backend lacks the updated multilayer "
            "compute_cypl.system(..., plane_thicknesses=...) interface. "
            "Select the updated checkout with CASIMIR_CYLINDER_ROOT."
        )
    _cylinder_system_factory = compute.system
    _normalize_plane_stack = plane.normalize_plane_stack
    fcqs_combined = quadratures.fcqs_combined
    fcqs_semiinfinite = quadratures.fcqs_semiinfinite
    _LOADED_BACKEND_REPO = selected_repo


class LayeredCylinderPlaneSystem:
    """Validated adapter over the upstream multilayer cylinder-plane system.

    Scattering, quadrature, frequency summation and the distinction between
    vacuum and medium wavenumbers are owned by ``compute_cypl.system``.
    """

    def __init__(self, temperature_k, distance_m, radius_m, mat_cylinder, plane_materials, plane_thicknesses_m, mat_medium):
        for name, value in (
            ("temperature_k", temperature_k),
            ("distance_m", distance_m),
            ("radius_m", radius_m),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        load_cylinder_backend()
        plane_materials = list(plane_materials)
        plane_thicknesses_m = list(plane_thicknesses_m)
        normalized_materials, normalized_thicknesses = _normalize_plane_stack(
            plane_materials, plane_thicknesses_m
        )
        finite_thicknesses = normalized_thicknesses if len(normalized_materials) > 1 else ()
        for thickness in finite_thicknesses:
            if thickness is None or not np.isfinite(thickness) or thickness <= 0.0:
                raise ValueError("Every finite plane-layer thickness must be finite and positive.")
        self.T = float(temperature_k)
        self.L = float(distance_m)
        self.R = float(radius_m)
        self.mat_cylinder = mat_cylinder
        self.plane_materials = plane_materials
        self.plane_thicknesses_m = plane_thicknesses_m
        self.mat_medium = mat_medium
        self._backend = _cylinder_system_factory(
            self.T,
            self.L,
            self.R,
            self.mat_cylinder,
            self.plane_materials,
            self.mat_medium,
            plane_thicknesses=self.plane_thicknesses_m,
        )
        self.x_quad = fcqs_combined
        self.y_quad = fcqs_semiinfinite
        self.n0_TM = None
        self.n0_TE = None
        self.n0 = None
        self.n1 = None

    def calculate(
        self,
        observable: str,
        *,
        ht_limit: bool = False,
        etaNx: float = 4.0,
        Nx: int | None = None,
        Ny: int = 40,
        etanmax: float = 10.0,
        nmax: int | None = None,
        fs: str = "psd",
        epsrel: float = 1e-8,
        N: int | None = None,
    ) -> float:
        if observable not in {"energy", "force", "forcegradient"}:
            raise ValueError("observable must be 'energy', 'force', or 'forcegradient'")

        if fs not in {"psd", "msd"}:
            raise ValueError("fs must be either 'psd' or 'msd'")

        if not np.isfinite(etaNx) or etaNx <= 0.0:
            raise ValueError("etaNx must be finite and positive.")
        if Nx is not None and (
            not isinstance(Nx, (int, np.integer))
            or isinstance(Nx, (bool, np.bool_))
            or Nx <= 0
            or Nx % 2 != 0
        ):
            raise ValueError("Nx must be a positive even integer or None for automatic selection.")
        if not isinstance(Ny, int) or isinstance(Ny, bool) or Ny <= 0:
            raise ValueError("Ny must be a positive integer.")
        if not np.isfinite(etanmax) or etanmax <= 0.0:
            raise ValueError("etanmax must be finite and positive.")
        if nmax is not None and (not isinstance(nmax, int) or isinstance(nmax, bool) or nmax <= 0):
            raise ValueError("nmax must be a positive integer or None for automatic selection.")
        if N is not None and (not isinstance(N, int) or isinstance(N, bool) or N <= 0):
            raise ValueError("summation order must be a positive integer or None for automatic selection.")
        if fs == "psd" and N is not None and N < 2:
            raise ValueError("casimir_cylinder requires PSD order >= 2.")
        if not np.isfinite(epsrel) or epsrel <= 0.0:
            raise ValueError("epsrel must be finite and positive.")

        backend = self._backend
        backend.T, backend.L, backend.R = self.T, self.L, self.R
        backend.mat_cylinder = self.mat_cylinder
        backend.mat_plane = self.plane_materials
        backend.plane_thicknesses = self.plane_thicknesses_m
        backend.mat_medium = self.mat_medium
        backend.x_quad, backend.y_quad = self.x_quad, self.y_quad
        # A later high-temperature-only call must not expose an earlier n1.
        backend.n1 = None
        self.n1 = None
        result = backend.calculate(
            observable,
            ht_limit=ht_limit,
            etaNx=etaNx,
            Nx=None if Nx is None else int(Nx),
            Ny=Ny,
            etanmax=etanmax,
            nmax=nmax,
            fs=fs,
            epsrel=epsrel,
            N=N,
        )
        for name in ("Nx", "Ny", "nmax", "n0_TM", "n0_TE", "n0", "n1"):
            setattr(self, name, getattr(backend, name, None))
        result = float(result)
        if not np.isfinite(result):
            raise FloatingPointError(f"Cylinder {observable} calculation returned a non-finite result.")
        return result


def load_material(module_name: str):
    ensure_cylinder_backend_source()
    module = importlib.import_module(f"casimir_cylinder.materials.{module_name}")
    module_file = getattr(module, "__file__", None)
    selected_package = CYLINDER_REPO.resolve() / "casimir_cylinder"
    if module_file is None or not Path(module_file).resolve().is_relative_to(selected_package):
        raise RuntimeError(
            f"Imported material {module_name!r} from outside the selected cylinder backend: "
            f"{module_file}"
        )
    if not hasattr(module, "epsilon") or not hasattr(module, "materialclass"):
        raise ValueError(f"Material module {module_name!r} must expose epsilon() and materialclass.")
    return module


__all__ = [
    "CYLINDER_REPO", "PROJECT_ROOT", "REQUIRED_MATERIAL_SET", "REQUIRED_COATING_MODEL",
    "REQUIRED_ROLE_MATERIALS", "LayeredCylinderPlaneSystem", "find_cylinder_repo",
    "ensure_cylinder_backend_source", "load_cylinder_backend", "load_material",
]
