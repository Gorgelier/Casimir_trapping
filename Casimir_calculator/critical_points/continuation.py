"""Fixed-thickness critical-locus queue, with explicit fixed-size advanced mode."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .evaluation import read_json, write_json
from .solver import solve_critical_point


def choose_seed(target, parameter, branch, rows, guesses):
    """Predict from computed points, falling back to an explicitly labelled guess."""
    known = [r for r in rows if r.get("status") == "converged"
             and (parameter == "size" or r.get("branch") == branch)]
    axis = "ptfe_thickness_nm" if parameter == "thickness" else "critical_size_nm"
    names = ("critical_size_nm", "critical_gap_nm") if parameter == "thickness" else ("ptfe_thickness_nm", "critical_gap_nm")
    if known:
        nearest = sorted(known, key=lambda r: abs(math.log(r[axis] / target)))
        a = nearest[0]
        b = next((r for r in nearest[1:] if abs(math.log(r[axis] / a[axis])) > 1e-5), None)
        z = np.log([a[name] for name in names])
        if b is not None:
            slope = (z - np.log([b[name] for name in names])) / math.log(a[axis] / b[axis])
            change = np.clip(slope * math.log(target / a[axis]), -0.5, 0.5)
            z += change
        return np.exp(z).tolist()
    candidates = [r for r in guesses if parameter == "size" or r.get("branch", "upper_size") == branch]
    if not candidates:
        raise ValueError(f"No explicit seed or computed point for {branch}")
    field = "thickness_nm" if parameter == "thickness" else "size_nm"
    a = min(candidates, key=lambda r: abs(math.log(r[field] / target)))
    return [a["size_nm"] if parameter == "thickness" else a["thickness_nm"], a["gap_nm"]]


class Continuation:
    def __init__(self, evaluator, directory, *, resume=False):
        self.evaluator = evaluator
        self.directory = Path(directory)
        self.resume = resume
        self.rows = []
        path = self.directory / "results.json"
        if path.exists():
            if not resume:
                raise FileExistsError("Results already exist; use --resume or a new output directory")
            old = read_json(path)
            if old.get("source_identity") != evaluator.identity or old.get("settings") != evaluator.settings:
                raise ValueError("Resume source/profile/settings mismatch; use a new output directory")
            self.rows = old["rows"]

    def save(self, status, **extra):
        write_json(self.directory / "results.json", {
            "schema_version": 1, "geometry": self.evaluator.geometry,
            "source_identity": self.evaluator.identity, "provenance": self.evaluator.provenance,
            "settings": self.evaluator.settings, "rows": self.rows,
            "progress": {"status": status, **extra},
            "scope": "Local finite-gap continuation; unresolved rows are not absence proofs."
        })

    def point(self, target, parameter="thickness", branch="upper_size", seed=None):
        if seed is None:
            seed = choose_seed(target, parameter, branch, self.rows,
                               self.evaluator.settings["seeds"][self.evaluator.geometry])
        free = "size" if parameter == "thickness" else "thickness"
        # Extrapolated guesses are clipped and genuinely corrected. They are
        # never promoted to solved roots merely because a predictor crossed a bound.
        seed = [float(np.clip(seed[0], *self.evaluator.settings["bounds_nm"][free])), seed[1]]
        token = f"{parameter}_{target:.12g}_{branch}".replace(".", "p")
        path = self.directory / "points" / (token + ".json")
        self.save("running", parameter=parameter, target_nm=target, branch=branch)
        # Repeated tasks within this queue may resume their own checkpoints.
        row = solve_critical_point(self.evaluator, target, seed, parameter=parameter,
                                   branch=branch, checkpoint=path, resume=self.resume or path.exists())
        key = (parameter, float(target), branch)
        self.rows = [r for r in self.rows if (r.get("parameter"), r.get("target_nm"), r.get("requested_branch")) != key]
        self.rows.append(row)
        self.save("running", last_status=row["status"])
        return row

    def run(self, thicknesses=None, sizes=None, *, max_points=None):
        settings = self.evaluator.settings
        geometry = self.evaluator.geometry
        guesses = settings["seeds"][geometry]
        queue = []
        if thicknesses is None and sizes is None:
            # Seed coordinates supply initial guesses only. They never add
            # tasks outside the requested fixed-thickness list.
            thicknesses = settings["sweep"]["thicknesses_nm"]
        if sizes is not None:
            queue.extend((float(s), "size", "upper_size", None) for s in sizes)
        if thicknesses is not None:
            upper_guesses = [g for g in guesses if g.get("branch", "upper_size") == "upper_size"]
            if not upper_guesses:
                raise ValueError("Fixed-thickness continuation requires an upper_size initial guess")
            ordered = sorted(thicknesses, key=lambda t: min(abs(math.log(t / g["thickness_nm"])) for g in upper_guesses))
            queue.extend((float(t), "thickness", "upper_size", None) for t in ordered)
        attempts = 0
        self.save("running", queued_tasks=len(queue))
        for target, parameter, branch, seed in queue:
            if any(r.get("parameter") == parameter and r.get("target_nm") == float(target)
                   and r.get("requested_branch") == branch and r.get("status") == "converged"
                   for r in self.rows):
                continue
            if max_points is not None and attempts >= max_points:
                self.save("budget_limited", attempted_this_run=attempts, queued_tasks=len(queue))
                return self.rows
            self.point(target, parameter, branch, seed)
            attempts += 1
        unresolved = sum(r["status"] != "converged" for r in self.rows)
        self.save("complete_with_unresolved" if unresolved else "complete",
                  unresolved_count=unresolved, attempted_this_run=attempts)
        return self.rows
