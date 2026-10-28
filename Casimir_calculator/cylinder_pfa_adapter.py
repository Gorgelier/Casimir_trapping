"""Cylinder energy per length from the teacher's layered-plane PFA code.

The teacher's integrands contain radius only as sqrt(R). At fixed separation,
temperature and materials we therefore evaluate one reference radius and reuse
that result by exact PFA scaling. This is not an interpolation in separation or
an approximation to the original scattering solver. PFA itself requires small
gap/radius; the adapter does not establish that validity or convergence.

Only energy is exposed. The third entry returned by upstream compute_pfa is a
placeholder zero and must never be reported as a calculated force gradient.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np

from . import cylinder_adapter

PFA_REFERENCE_RADIUS_M = 1.0e-6
PFA_NUMERICS = {
    "frequency_summation": "psd",
    "psd_epsrel": 1.0e-8,
    "psd_order": "upstream automatic",
    "radial_quadrature": "scipy.integrate.quad defaults",
    "polylog_quadrature": "scipy.integrate.quad defaults where used upstream",
    "quad_epsabs": 1.49e-8,
    "quad_epsrel": 1.49e-8,
    "quad_limit": 50,
    "reference_radius_m": PFA_REFERENCE_RADIUS_M,
    "radius_scaling": "sqrt(radius_m / reference_radius_m)",
    "energy_units": "J/m",
    "forcegradient_available": False,
}


def _finite_real(value, name: str, *, positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return result


def _source_fingerprint(path) -> tuple[str, str]:
    source = Path(path).resolve()
    return str(source), hashlib.sha256(source.read_bytes()).hexdigest()


def _load_pfa_backend():
    """Verify the selected checkout without importing compute_cypl.system."""
    cylinder_adapter.ensure_cylinder_backend_source()
    selected = cylinder_adapter.CYLINDER_REPO.resolve()
    modules = tuple(importlib.import_module(f"casimir_cylinder.{name}") for name in
                    ("pfa_cypl", "plane", "frequency_summation", "psd"))
    sources = []
    for module in modules:
        filename = getattr(module, "__file__", None)
        if filename is None or not Path(filename).resolve().is_relative_to(selected / "casimir_cylinder"):
            raise RuntimeError(f"Imported {module.__name__} from outside the selected PFA checkout: {filename}")
        sources.append(_source_fingerprint(filename))
    compute = getattr(modules[0], "compute_pfa", None)
    normalize = getattr(modules[1], "normalize_plane_stack", None)
    if not callable(compute) or "thicknesses" not in inspect.signature(compute).parameters or not callable(normalize):
        raise RuntimeError("The selected teacher PFA backend requires compute_pfa(..., thicknesses=...).")
    # Callable identity isolates reloaded or replaced functions as well as roots.
    return compute, normalize, (str(selected), tuple(sources), normalize)


def _material_source_token(material) -> tuple:
    """Keep material identities and the actual reference-model provenance.

    Materials must be hashable and must not be mutated while cached. The
    application's frozen material adapters satisfy this contract. Changing the
    reference-material cache changes its recorded loaded-source hashes too.
    """
    try:
        hash(material)
    except TypeError as exc:
        raise ValueError("PFA materials must be hashable, immutable material objects.") from exc
    epsilon = getattr(material, "epsilon", None)
    if not callable(epsilon) or getattr(material, "materialclass", None) not in {"dielectric", "drude", "plasma", "PEC"}:
        raise ValueError("PFA materials require epsilon(xi) and a supported materialclass.")
    permittivity = getattr(material, "permittivity", epsilon)
    paths = set()
    for obj in (material if isinstance(material, ModuleType) else type(material), permittivity):
        try:
            filename = inspect.getsourcefile(obj)
        except (TypeError, OSError):
            filename = getattr(obj, "__file__", None)
        if filename and Path(filename).is_file():
            paths.add(str(Path(filename).resolve()))
    reference_provenance = None
    if getattr(permittivity, "__module__", "") == "Casimir_calculator.reference_materials":
        from .reference_materials import cylinder_reference_provenance
        reference_provenance = json.dumps(cylinder_reference_provenance(), sort_keys=True, allow_nan=False)
    return tuple(_source_fingerprint(path) for path in sorted(paths)), reference_provenance


@lru_cache(maxsize=16384)
def _reference_energy(compute, backend_token, material_tokens, temperature_k,
                      distance_m, mat_cylinder, plane_materials,
                      plane_thicknesses_m, mat_medium):
    n0, n1 = compute(temperature_k, distance_m, PFA_REFERENCE_RADIUS_M,
                     mat_cylinder, list(plane_materials), mat_medium,
                     thicknesses=list(plane_thicknesses_m))
    contributions = []
    for name, contribution in (("n0", n0), ("n1", n1)):
        if np.iscomplexobj(contribution):
            raise ValueError(f"Teacher PFA returned complex {name}.")
        array = np.asarray(contribution, dtype=float)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError(f"Teacher PFA returned invalid or non-finite {name}.")
        contributions.append(float(array[0]))
    result = sum(contributions)
    if not math.isfinite(result):
        raise ValueError("Teacher PFA returned a non-finite total energy.")
    return result


def pfa_energy_per_length(temperature_k, distance_m, radius_m, mat_cylinder,
                          plane_materials, plane_thicknesses_m, mat_medium) -> float:
    """Return (n0[0] + n1[0]) in J/m with the teacher's unchanged numerics.

    Floats in the cache key are never rounded. Radius is intentionally excluded
    from that key because the analytical PFA factor is applied after lookup.
    A zero-radius boundary belongs to the sweep's degenerate-geometry handling.
    """
    temperature_k = _finite_real(temperature_k, "temperature_k", positive=True)
    distance_m = _finite_real(distance_m, "distance_m", positive=True)
    radius_m = _finite_real(radius_m, "radius_m", positive=True)
    compute, normalize, backend_token = _load_pfa_backend()
    materials, thicknesses = normalize(plane_materials, plane_thicknesses_m)
    materials = tuple(materials)
    thicknesses = ((None,) if len(materials) == 1 else tuple(
        _finite_real(value, "plane-layer thickness", positive=False) for value in thicknesses
    ))
    material_tokens = tuple(_material_source_token(material) for material in
                            (mat_cylinder, *materials, mat_medium))
    reference = _reference_energy(compute, backend_token, material_tokens,
                                  temperature_k, distance_m, mat_cylinder,
                                  materials, thicknesses, mat_medium)
    result = reference * math.sqrt(radius_m / PFA_REFERENCE_RADIUS_M)
    if not math.isfinite(result):
        raise ValueError("Radius-scaled PFA energy is non-finite.")
    return result


def pfa_cache_info():
    """Return the reference-energy LRU cache's hit/miss statistics."""
    return _reference_energy.cache_info()


def clear_pfa_cache() -> None:
    """Discard PFA energies; reference material models have their own cache."""
    _reference_energy.cache_clear()


__all__ = ["pfa_energy_per_length", "pfa_cache_info", "clear_pfa_cache",
           "PFA_REFERENCE_RADIUS_M", "PFA_NUMERICS"]
