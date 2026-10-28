"""Call the professor's Au/ethanol/PTFE models from the selected checkout.

No oscillator coefficients are copied here. Source modules are loaded in an
isolated namespace so these lightweight material calls neither import the full
cylinder solver nor interfere with its choice of checkout in ``sys.modules``.
All frequencies are imaginary angular frequencies xi in rad/s.
"""

from __future__ import annotations

from functools import lru_cache
from collections.abc import Mapping
import hashlib
import os
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_SOURCE_NAMES = ("gold_real", "ethanol", "teflon")
# epsilon(0) diverges for Drude gold. Local plane Fresnel code needs a finite
# value to avoid inf * 0 in kappa. Only exact zero is replaced during material
# evaluation: optics still receives xi=0, hence kappa=k_parallel, TE=0 and TM=1
# (to floating precision). This is not a fitted low-frequency cutoff.
_GOLD_ZERO_PROXY_XI = 1.0e-16


def cylinder_material_source() -> Path:
    """Select the bundled teacher checkout; an environment override is final."""
    return _resolve_material_source(os.environ.get("CASIMIR_CYLINDER_ROOT"))


@lru_cache(maxsize=8)
def _resolve_material_source(override: str | None) -> Path:
    # Scalar scattering quadratures call epsilon very frequently. Resolve and
    # validate once per selected root, rather than issue filesystem queries on
    # every material sample. A changed override has a distinct cache key.
    source = (
        Path(override).expanduser()
        if override is not None
        else Path(__file__).resolve().parents[1] / "external" / "casimir_cylinder-main"
    ).resolve()
    if override is not None and not override.strip():
        raise RuntimeError("CASIMIR_CYLINDER_ROOT must not be empty.")
    if not (source / "casimir_cylinder" / "materials").is_dir():
        raise RuntimeError(
            "The selected cylinder material checkout must contain "
            f"casimir_cylinder/materials: {source}. Set CASIMIR_CYLINDER_ROOT "
            "to a complete casimir_cylinder-main checkout."
        )
    return source


@lru_cache(maxsize=8)
def _load_source_modules(source_text: str) -> tuple[dict[str, ModuleType], dict[str, str]]:
    source = Path(source_text)
    modules: dict[str, ModuleType] = {}
    hashes: dict[str, str] = {}
    for name in _SOURCE_NAMES:
        source_file = (source / "casimir_cylinder" / "materials" / f"{name}.py").resolve()
        if not source_file.is_relative_to(source):
            raise RuntimeError(f"Material source resolves outside the selected checkout: {source_file}")
        if not source_file.is_file():
            raise RuntimeError(f"Required cylinder material source is missing: {source_file}")
        source_bytes = source_file.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        module = ModuleType(f"_casimir_reference_{name}_{digest[:16]}")
        module.__file__ = str(source_file)
        # Compile exactly the bytes whose digest is recorded, avoiding stale
        # bytecode caches. __name__ also prevents upstream plotting examples.
        exec(compile(source_bytes, str(source_file), "exec"), module.__dict__)
        if not callable(getattr(module, "epsilon", None)):
            raise RuntimeError(f"Cylinder material source has no callable epsilon(xi): {source_file}")
        expected_class = "drude" if name == "gold_real" else "dielectric"
        if getattr(module, "materialclass", None) != expected_class:
            raise RuntimeError(f"Unexpected materialclass in cylinder material source: {source_file}")
        modules[name] = module
        hashes[name] = digest
    return modules, hashes


def _evaluate_reference(name: str, xi: float | np.ndarray) -> np.ndarray:
    if np.iscomplexobj(xi):
        raise ValueError("xi must be a real, finite, non-negative angular frequency in rad/s.")
    xi_array = np.asarray(xi, dtype=float)
    if not np.all(np.isfinite(xi_array)) or np.any(xi_array < 0.0):
        raise ValueError("xi must be finite and non-negative, in rad/s.")
    modules, _ = _load_source_modules(str(cylinder_material_source()))
    epsilon = modules[name].epsilon
    flat_xi = xi_array.reshape(-1)
    if name == "gold_real":
        flat_xi = np.where(flat_xi == 0.0, _GOLD_ZERO_PROXY_XI, flat_xi)
    # Upstream ethanol branches on scalar xi and teflon sums over its oscillator
    # axis. Scalar dispatch preserves their exact behavior for arbitrary input
    # shapes, including arrays whose length equals their oscillator count.
    values = np.fromiter((float(epsilon(float(value))) for value in flat_xi), dtype=float, count=flat_xi.size)
    if not np.all(np.isfinite(values)) or np.any(values < 1.0):
        raise ValueError(f"The selected {name}.py produced invalid epsilon(i xi) values.")
    return values.reshape(xi_array.shape)


def gold_cylinder_reference_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Teacher gold_real.py: Drude plus four fitted Lorentz oscillators.

    The supplied fit's complete optical-data provenance is not documented in
    the source. Exact zero uses a finite metallic proxy; see module comments.
    """
    return _evaluate_reference("gold_real", xi)


def ethanol_cylinder_reference_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Teacher ethanol.py: seven oscillators with exact epsilon(0)=24.3."""
    return _evaluate_reference("ethanol", xi)


def teflon_cylinder_reference_permittivity(xi: float | np.ndarray) -> np.ndarray:
    """Teacher teflon.py: Zwol/Palasantzas eight-oscillator ordinary PTFE."""
    return _evaluate_reference("teflon", xi)


def cylinder_reference_provenance(
    *, material_set: str = "cylinder-reference", coating_model: str = "teflon-cylinder-reference",
) -> dict[str, Any]:
    """Describe the source bytes actually loaded by this process, not a fit copy.

    Sources are cached for the run. After editing upstream files in the same
    process, call clear_reference_material_cache() before recalculating.
    """
    selected_names = _reference_source_names(material_set, coating_model)
    result: dict[str, Any] = {
        "material_set": material_set, "coating_model": coating_model,
        "frequency_units": "rad/s", "sources": {},
    }
    if not selected_names:
        return result
    source = cylinder_material_source()
    modules, hashes = _load_source_modules(str(source))
    result["root"] = str(source)
    result["sources"] = {
        name: {"file": str(modules[name].__file__), "sha256": hashes[name]}
        for name in _SOURCE_NAMES if name in selected_names
    }
    result["dielectric_reference"] = "P. J. van Zwol and G. Palasantzas, Phys. Rev. A 81, 062502 (2010); DOI: 10.1103/PhysRevA.81.062502"
    if "gold_real" in selected_names:
        result.update({
            "gold_model": "gold_real.py; Drude plus four fitted Lorentz oscillators",
            "gold_source_note": "Full fitted optical-data provenance is not documented in gold_real.py.",
            "gold_zero_mode": "Drude; local Fresnel epsilon(0) uses gold_real.epsilon(1e-16 rad/s), while optical xi remains exactly zero",
            "ethanol_epsilon_zero": float(modules["ethanol"].epsilon(0.0)),
        })
    if "teflon" in selected_names:
        result.update({
            "coating_material": "ordinary PTFE; not Teflon AF2400",
            "teflon_epsilon_zero": float(modules["teflon"].epsilon(0.0)),
        })
    return result


def clear_reference_material_cache() -> None:
    """Reload source modules on the next call after a deliberate source edit."""
    _load_source_modules.cache_clear()
    _resolve_material_source.cache_clear()


def uses_reference_materials(material_set: object, coating_model: object) -> bool:
    """Whether either active model uses the teacher source checkout.

    The historical literal ``default`` is deliberately not a reference alias.
    """
    return material_set == "cylinder-reference" or coating_model == "teflon-cylinder-reference"


def _reference_source_names(material_set: object, coating_model: object) -> set[str]:
    names: set[str] = set()
    if material_set == "cylinder-reference":
        names.update(("gold_real", "ethanol"))
    if coating_model == "teflon-cylinder-reference":
        names.add("teflon")
    return names


def extract_reference_source_hashes(
    payload: object, *, required: bool = False,
    material_set: object = None, coating_model: object = None,
) -> dict[str, str] | None:
    """Extract complete, source-only model identity from saved run metadata.

    Accept a direct provenance object returned by cylinder_reference_provenance,
    a direct ``{model_name: sha256}`` mapping, known metadata wrappers, or a
    list/tuple of payloads from multiple sidecars. Only
    these wrappers are traversed (at most eight levels and 128 mappings), not
    arbitrary result arrays or unrelated source-fingerprint objects. Every
    discovered record must contain all selected teacher models; without explicit
    model labels all three are required. Records with different hashes are
    rejected. Checkout locations and filenames do not affect model identity.
    ``required=True`` rejects metadata with no required reference source record.
    Explicit models using no teacher sources return None.
    """
    wrappers = (
        "reference_material_sources", "material_sources", "material_reference",
        "material_source_provenance", "model_metadata", "config", "system",
        "physical_model", "reference_source_hashes", "material_source_hashes",
        "material_reference_source_hashes", "native_model", "convergence_metadata",
        "backend", "base", "refined", "configuration",
    )
    model_names = set(_SOURCE_NAMES)
    expected_names = (
        model_names if material_set is None and coating_model is None
        else _reference_source_names(material_set, coating_model)
    )
    if not expected_names:
        return None
    selected: dict[str, str] | None = None
    visited: set[int] = set()

    def parse_record(record: Mapping) -> dict[str, str]:
        if not expected_names.issubset(record) or not set(record).issubset(model_names):
            raise ValueError("Reference material source hashes must contain the complete selected sources: " + ", ".join(sorted(expected_names)) + ".")
        hashes: dict[str, str] = {}
        for name in _SOURCE_NAMES:
            if name not in expected_names:
                continue
            entry = record[name]
            value = entry.get("sha256") if isinstance(entry, Mapping) else entry
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
                raise ValueError(f"Reference material {name} must have a valid SHA256 source hash.")
            hashes[name] = value.lower()
        return hashes

    explicit_source_wrappers = {
        "reference_material_sources", "material_reference", "material_source_provenance",
        "reference_source_hashes", "material_source_hashes", "material_reference_source_hashes",
    }

    def inspect(node: object, depth: int = 0, reference_context: bool = False) -> None:
        nonlocal selected
        if node is None:
            return
        if isinstance(node, (list, tuple)):
            if id(node) in visited:
                return
            if depth > 8 or len(node) > 128 or len(visited) >= 128:
                raise ValueError("Reference material provenance nesting exceeds the supported limit.")
            visited.add(id(node))
            for item in node:
                inspect(item, depth + 1, reference_context)
            return
        if not isinstance(node, Mapping):
            raise ValueError("Reference material provenance must be a JSON object.")
        if id(node) in visited:
            return
        if depth > 8 or len(visited) >= 128:
            raise ValueError("Reference material provenance nesting exceeds the supported limit.")
        visited.add(id(node))
        record: Mapping | None = None
        if model_names.intersection(node):
            record = node
        elif "sources" in node:
            sources = node["sources"]
            reference_context = reference_context or uses_reference_materials(node.get("material_set"), node.get("coating_model"))
            if not isinstance(sources, Mapping) and reference_context:
                raise ValueError("Reference material sources must be a JSON object.")
            if isinstance(sources, Mapping) and (
                reference_context or not sources or model_names.intersection(sources)
            ):
                record = sources
        if record is not None:
            hashes = parse_record(record)
            if selected is not None and any(selected[name] != value for name, value in hashes.items() if name in selected):
                raise ValueError("Conflicting reference material source hashes in saved metadata.")
            selected = {**(selected or {}), **hashes}
        for key in wrappers:
            if key in node:
                inspect(node[key], depth + 1, key in explicit_source_wrappers)

    if payload is not None:
        inspect(payload)
    if selected is None and required:
        raise ValueError("Reference material source hashes are required but missing from saved metadata.")
    return {name: selected[name] for name in _SOURCE_NAMES if name in expected_names} if selected is not None else None


__all__ = [
    "clear_reference_material_cache",
    "cylinder_material_source",
    "cylinder_reference_provenance",
    "extract_reference_source_hashes",
    "ethanol_cylinder_reference_permittivity",
    "gold_cylinder_reference_permittivity",
    "teflon_cylinder_reference_permittivity",
    "uses_reference_materials",
]
