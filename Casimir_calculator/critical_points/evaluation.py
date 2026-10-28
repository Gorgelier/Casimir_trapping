"""Cached force evaluations, explicit units, budgets, and source provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np

from .settings import canonical_geometry


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_provenance(geometry):
    """Content identity includes selected sources and numerical dependencies.

    Path labels within bundled trees remain portable. Override roots are recorded
    as metadata; their content, not the machine-specific location, keys caches.
    """
    from ..reference_materials import cylinder_material_source
    root = Path(__file__).resolve().parents[2]
    trees = {"Casimir_calculator": root / "Casimir_calculator",
             "cylinder_materials": cylinder_material_source() / "casimir_cylinder" / "materials"}
    if geometry == "cylinder":
        from ..cylinder_adapter import CYLINDER_REPO
        trees["cylinder_backend"] = CYLINDER_REPO / "casimir_cylinder"
    if geometry == "sphere":
        from ..caspwn_adapter import caspwn_source
        trees["sphere_backend"] = caspwn_source() / "caspwn"
    hashes = {}
    for label, directory in trees.items():
        for path in sorted(directory.rglob("*.py")):
            hashes[f"{label}/{path.relative_to(directory).as_posix()}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "numba"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"sources": hashes, "dependencies": versions,
            "selected_roots": {k: str(v) for k, v in trees.items()}}


class BudgetExceeded(RuntimeError):
    pass


class Evaluator:
    def __init__(self, geometry, settings, cache_dir, backend=None):
        self.geometry = canonical_geometry(geometry)
        self.settings = settings
        self.cache_dir = Path(cache_dir)
        self.started = time.monotonic()
        self.fresh = self.hits = 0
        if backend is None:
            from .plate import PlateForce
            from .cylinder import CylinderForce
            from .sphere import SphereForce
            backend = {"plane": PlateForce, "cylinder": CylinderForce, "sphere": SphereForce}[self.geometry](settings)
            self.provenance = source_provenance(self.geometry)
        else:
            self.provenance = {"sources": {"synthetic": str(type(backend))}, "dependencies": {}, "selected_roots": {}}
        self.backend = backend
        self.identity = digest({"geometry": self.geometry, "physics": settings["physics"],
                                "profile": settings["profiles"][self.geometry],
                                "sources": self.provenance["sources"], "dependencies": self.provenance["dependencies"]})
        self.memory = {}

    def reset_budget(self):
        self.started = time.monotonic()
        self.fresh = self.hits = 0

    def budget(self):
        limits = self.settings["solver"]
        if self.fresh >= limits["max_evaluations"]:
            raise BudgetExceeded("force-evaluation budget reached")
        if time.monotonic() - self.started >= limits["max_seconds"]:
            raise BudgetExceeded("wall-time budget reached between backend calls")

    def load(self, size_nm):
        p = self.settings["physics"]
        gamma = (p["gold_density_kg_m3"] - p["ethanol_density_kg_m3"]) * p["gravity_m_s2"]
        s = size_nm * 1e-9
        weight = gamma * {"plane": s, "cylinder": math.pi * s**2, "sphere": 4 * math.pi * s**3 / 3}[self.geometry]
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Outward buoyancy-corrected load must be finite and positive")
        return weight

    def validate_coordinates(self, t, s, d):
        for name, value in zip(("thickness", "size", "gap"), (t, s, d)):
            lo, hi = self.settings["bounds_nm"][name]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name}={value} nm lies outside [{lo}, {hi}]")

    def force(self, t, s, d):
        self.validate_coordinates(t, s, d)
        spec = {"identity": self.identity, "t_nm": float(t), "s_nm": float(s), "d_nm": float(d)}
        key = digest(spec)
        if key in self.memory:
            self.hits += 1
            return self.memory[key]
        path = self.cache_dir / "samples" / (key + ".json")
        if path.exists():
            packet = read_json(path)
            if packet["spec"] != spec or digest(packet["sample"]) != packet["sample_sha256"]:
                raise ValueError("Force cache checksum or settings mismatch")
            sample = packet["sample"]
            self.hits += 1
        else:
            self.budget()
            self.fresh += 1
            started = time.monotonic()
            force, gradient = self.backend(t, s, d)
            if not math.isfinite(force) or (gradient is not None and not math.isfinite(gradient)):
                raise ArithmeticError("Non-finite scattering result")
            sample = {"casimir_force": float(force), "native_gradient_per_nm": gradient,
                      "force_unit": self.backend.force_unit, "seconds": time.monotonic() - started}
            write_json(path, {"spec": spec, "sample": sample, "sample_sha256": digest(sample)})
        self.memory[key] = sample
        return sample

    def stencil(self, t, s, d, step=None):
        lo, hi = self.settings["bounds_nm"]["gap"]
        step = min(0.05, max(0.01, d * 0.0005)) if step is None else step
        step = min(step, (d - lo) * 0.45, (hi - d) * 0.45)
        if step <= max(1e-8, abs(d) * 1e-10):
            raise ValueError("Gap stencil requires an interior gap")
        left, center, right = [self.force(t, s, d + j * step) for j in (-1, 0, 1)]
        fm, f, fp = [x["casimir_force"] for x in (left, center, right)]
        actual_fd = (fp - fm) / (2 * step)
        if self.backend.native_gradient:
            fd = center["native_gradient_per_nm"]
            curvature = (right["native_gradient_per_nm"] - left["native_gradient_per_nm"]) / (2 * step)
        else:
            fd = actual_fd
            curvature = (fp - 2 * f + fm) / step**2
        weight = self.load(s)
        return {"t": float(t), "s": float(s), "d": float(d), "f": f + weight,
                "fd": fd, "actual_fd": actual_fd, "curvature": curvature,
                "weight": weight, "step_nm": step,
                "residual": np.array([(f + weight) / weight, d * fd / weight])}

    def size_derivative(self, t, s, d):
        lo, hi = self.settings["bounds_nm"]["size"]
        h = max(0.01, s * 2e-4)
        a, b = max(lo, s - h), min(hi, s + h)
        fa = self.force(t, a, d)["casimir_force"] + self.load(a)
        fb = self.force(t, b, d)["casimir_force"] + self.load(b)
        return (fb - fa) / (b - a)
