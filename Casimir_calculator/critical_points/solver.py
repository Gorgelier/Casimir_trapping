"""Bounded log-coordinate Newton/Broyden correction for F = F_d = 0."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .evaluation import BudgetExceeded, digest, read_json, write_json


def passed(q, limits):
    c = q["curvature"]
    return (c > 0 and abs(q["residual"][0]) <= limits["force_relative_tolerance"]
            and abs(q["residual"][1]) <= limits["gradient_scaled_tolerance"]
            and abs(q["fd"] / c) <= limits["stationarity_tolerance_nm"]
            and abs(q["actual_fd"] / c) <= limits["stationarity_tolerance_nm"])


def solve_critical_point(evaluator, target, seed, *, parameter="thickness",
                         branch="upper_size", checkpoint=None, resume=False):
    """Solve at fixed thickness or fixed size, with seed=(free coordinate, gap).

    A failed or budget-limited local search is unresolved, not an absence proof.
    The elapsed-time budget is checked between native evaluations; a running
    external solver call is not interrupted. Use a distinct output per process.
    """
    if parameter not in ("thickness", "size"):
        raise ValueError("parameter must be thickness or size")
    settings = evaluator.settings
    limits = settings["solver"]
    free = "size" if parameter == "thickness" else "thickness"
    fixed_bounds = settings["bounds_nm"][parameter]
    if not np.isfinite(target) or not fixed_bounds[0] <= target <= fixed_bounds[1]:
        raise ValueError(f"Fixed {parameter} is outside the requested bounds")
    lower = np.log([settings["bounds_nm"][free][0], settings["bounds_nm"]["gap"][0]])
    upper = np.log([settings["bounds_nm"][free][1], settings["bounds_nm"]["gap"][1]])
    task = {"geometry": evaluator.geometry, "parameter": parameter, "target_nm": float(target),
            "requested_branch": branch, "source_identity": evaluator.identity,
            "bounds_nm": settings["bounds_nm"], "solver": limits}
    task_identity = digest(task)
    path = Path(checkpoint) if checkpoint is not None else None
    history = []
    if path is not None and path.exists():
        if not resume:
            raise FileExistsError(f"Checkpoint exists; use --resume or a new output: {path}")
        old = read_json(path)
        if old.get("task_identity") != task_identity:
            raise ValueError("Checkpoint physics, source, bounds, or solver settings differ; choose a new output")
        if old.get("status") == "converged":
            return old
        history = old.get("iterations", [])
        if old.get("resume_seed") is not None:
            seed = old["resume_seed"]
        elif history:
            last = history[-1]
            seed = [last["size_nm"] if parameter == "thickness" else last["thickness_nm"], last["gap_nm"]]
    evaluator.reset_budget()
    started = time.monotonic()
    base = {"geometry": evaluator.geometry, "parameter": parameter, "target_nm": float(target),
            "ptfe_thickness_nm": float(target) if parameter == "thickness" else None,
            "critical_size_nm": float(target) if parameter == "size" else None,
            "critical_gap_nm": None, "branch": branch, "requested_branch": branch,
            "task_identity": task_identity, "source_identity": evaluator.identity,
            "profile": settings["profiles"][evaluator.geometry],
            "force_unit": evaluator.backend.force_unit, "native_gradient_used": evaluator.backend.native_gradient,
            "PFA_used": False, "iterations": history}

    def persist(status, reason=None, **extra):
        row = {**base, "status": status, "elapsed_seconds": time.monotonic() - started,
               "fresh_evaluations": evaluator.fresh, "cache_hits": evaluator.hits, **extra}
        if reason:
            row["reason"] = reason
        if status.startswith("unresolved"):
            row["note"] = "Local corrector did not certify a root; this is not evidence of absence."
        if path is not None:
            write_json(path, row)
        return row

    try:
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (2,) or not np.isfinite(seed).all() or np.any(seed <= 0):
            raise ValueError("Seed must contain two finite positive coordinates")
        z = np.log(seed)
        if np.any(z < lower) or np.any(z > upper):
            return persist("unresolved_seed_outside_bounds", "Initial guess outside requested domain")
        jacobian = None
        previous = None

        def at(position, step=None):
            # Avoid exp(log(bound)) roundoff crossing a hard domain boundary.
            p, d = np.exp(np.clip(position, lower, upper))
            p = float(np.clip(p, *settings["bounds_nm"][free]))
            d = float(np.clip(d, *settings["bounds_nm"]["gap"]))
            t, s = (target, p) if parameter == "thickness" else (p, target)
            return evaluator.stencil(float(t), float(s), d, step)

        for iteration in range(int(limits["max_iterations"])):
            evaluator.budget()
            q = at(z)
            y = q["residual"]
            history.append({"iteration": len(history), "thickness_nm": q["t"], "size_nm": q["s"],
                            "gap_nm": q["d"], "force_relative": float(y[0]),
                            "gradient_scaled": float(y[1]), "curvature_per_nm2": q["curvature"],
                            "stationarity_error_nm": q["fd"] / q["curvature"] if q["curvature"] else None})
            persist("running")
            if passed(q, limits):
                half = at(z, q["step_nm"] / 2)
                if passed(half, limits):
                    fs = evaluator.size_derivative(half["t"], half["s"], half["d"])
                    # F_s near zero marks a turn of the thickness locus. Such a
                    # point solves F=F_d=0 but is not certified as a simple fold in s.
                    transverse = abs(fs * half["s"] / half["weight"]) > max(1e-6, 10 * abs(half["residual"][0]))
                    measured_branch = "upper_size" if fs > 0 else "lower_size" if fs < 0 else "turning"
                    return persist("converged", ptfe_thickness_nm=half["t"], critical_size_nm=half["s"],
                                   critical_gap_nm=half["d"], branch=measured_branch,
                                   in_size_domain=True, nondegenerate_in_size=transverse,
                                   branch_matches_request=measured_branch == branch,
                                   residuals={"force_relative": float(half["residual"][0]),
                                              "gradient_scaled": float(half["residual"][1]),
                                              "stationarity_error_nm": half["fd"] / half["curvature"],
                                              "actual_force_derivative_error_nm": half["actual_fd"] / half["curvature"],
                                              "Fdd_native_per_nm2": half["curvature"],
                                              "Fs_native_per_nm": fs,
                                              "normalized_size_transversality": fs * half["s"] / half["weight"],
                                              "half_step_nm": half["step_nm"]})
            if previous is not None and jacobian is not None:
                dz = z - previous[0]
                dy = y - previous[1]
                if float(dz @ dz) > 1e-20:
                    jacobian += np.outer(dy - jacobian @ dz, dz) / float(dz @ dz)
            if jacobian is None or iteration % int(limits["jacobian_refresh_iterations"]) == 0:
                za, zb = z.copy(), z.copy()
                za[0] = max(lower[0], z[0] - 0.001)
                zb[0] = min(upper[0], z[0] + 0.001)
                qa, qb = at(za), at(zb)
                first = (qb["residual"] - qa["residual"]) / (zb[0] - za[0])
                second = np.array([q["fd"] * q["d"] / q["weight"],
                                   (q["fd"] + q["d"] * q["curvature"]) * q["d"] / q["weight"]])
                jacobian = np.column_stack([first, second])
            if not np.isfinite(jacobian).all() or np.linalg.cond(jacobian) > 1e13:
                return persist("unresolved_singular_jacobian", "Try fixed-size continuation near a thickness turning point")
            step = np.linalg.solve(jacobian, -y)
            step /= max(1.0, float(np.max(abs(step))) / limits["max_log_step"])
            candidate = np.clip(z + step, lower, upper)
            if np.linalg.norm(candidate - z) < 1e-11:
                return persist("unresolved_domain_boundary", "Corrector stalled at requested bounds")
            previous = (z.copy(), y.copy())
            z = candidate
            base["resume_seed"] = np.exp(z).tolist()
        return persist("unresolved_iteration_budget", "Iteration budget reached")
    except BudgetExceeded as exc:
        return persist("unresolved_budget", str(exc))
    except (ArithmeticError, ValueError, RuntimeError) as exc:
        return persist("unresolved_evaluation", f"{type(exc).__name__}: {exc}")
