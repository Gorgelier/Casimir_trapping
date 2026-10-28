"""One resumable size scan, shared by the three supported geometries."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

GEOMETRIES = ("plane", "cylinder", "sphere")
LIMITS_NM = {"plane": 1000.0, "cylinder": 1000.0, "sphere": 1000.0}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = {
    "schema_version": 1,
    "physics": {
        "temperature_k": 300.0, "teflon_thickness_nm": 60.0,
        "object_density_kg_m3": 19320.0, "medium_density_kg_m3": 785.06,
        "gravity_m_s2": 9.81,
    },
    "scans": {
        "plane": {"minimum_nm": 10.0, "maximum_nm": 1000.0, "spacing": "log", "points": 36},
        "cylinder": {"minimum_nm": 10.0, "maximum_nm": 1000.0, "spacing": "log", "points": 36},
        "sphere": {"minimum_nm": 10.0, "maximum_nm": 1000.0, "spacing": "log", "points": 36},
    },
    "separation": {
        "minimum_nm": 1.0, "maximum_nm": 1000.0, "points": 81,
        "max_extensions": 3, "extension_factor": 2.0,
    },
    "numerics": {
        "plane": {"fs": "psd", "frequency_order": None, "epsrel": 1e-6, "epsabs": 0.0},
        "cylinder": {"backend": "exact", "Nx": None, "Ny": 40, "nmax": None, "fs": "psd", "frequency_order": None, "epsrel": 1e-6},
        "sphere": {"n_radial": None, "m_angular": None, "psd_order": None, "epsrel": 1e-6,
                   "lmax_factor": None, "min_lmax": None, "cores": 1},
    },
    "analysis": {"refine_extrema": True, "max_refine_evaluations": 32, "grid_convergence": False},
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json(value))
    temporary.replace(path)


def _number(value: Any, name: str, *, minimum: float = 0.0, strict: bool = False, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name}: expected a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name}: expected a finite number")
    if value < minimum or (strict and value == minimum) or (integer and type(value) is not int):
        raise ValueError(f"{name}: invalid value {value!r}")
    return number


def validate_settings(settings: dict[str, Any]) -> None:
    def shape(actual: Any, template: Any, path: str) -> None:
        if isinstance(template, dict):
            if path == "settings.numerics.cylinder" and isinstance(actual, dict) and "backend" not in actual:
                # Older saved scans used exact scattering; accepting their schema
                # must not silently switch their physical approximation.
                template = {key: value for key, value in template.items() if key != "backend"}
            if path in {f"settings.scans.{geometry}" for geometry in GEOMETRIES}:
                return  # Size-grid schemas are validated by size_grid below.
            if not isinstance(actual, dict) or actual.keys() != template.keys():
                raise ValueError(f"{path}: expected exactly these keys: {', '.join(template)}")
            for key in template:
                shape(actual[key], template[key], f"{path}.{key}")
    shape(settings, DEFAULT_SETTINGS, "settings")
    if type(settings["schema_version"]) is not int or settings["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for key, value in settings["physics"].items():
        _number(value, f"physics.{key}", strict=key not in {"teflon_thickness_nm", "medium_density_kg_m3"})
    for geometry, spec in settings["scans"].items():
        grid = size_grid(spec)
        if grid[-1] > LIMITS_NM[geometry]:
            raise ValueError(f"{geometry}: require 0 <= minimum < maximum <= {LIMITS_NM[geometry]:g} nm")
    sep = settings["separation"]
    lo = _number(sep["minimum_nm"], "separation.minimum_nm", strict=True)
    hi = _number(sep["maximum_nm"], "separation.maximum_nm", strict=True)
    if hi <= lo:
        raise ValueError("separation.maximum_nm must exceed minimum_nm")
    _number(sep["points"], "separation.points", minimum=9, integer=True)
    _number(sep["max_extensions"], "separation.max_extensions", minimum=1, integer=True)
    _number(sep["extension_factor"], "separation.extension_factor", minimum=1, strict=True)
    for geometry, numeric in settings["numerics"].items():
        for key, value in numeric.items():
            if key == "backend":
                if value not in {"pfa", "exact"}:
                    raise ValueError("numerics.cylinder.backend must be pfa or exact")
            elif key == "fs":
                if value not in {"psd", "msd"}:
                    raise ValueError(f"numerics.{geometry}.fs must be psd or msd")
            elif value is None:
                if key in {"Ny", "cores", "epsrel", "epsabs"}:
                    raise ValueError(f"numerics.{geometry}.{key} must be numeric")
            else:
                _number(value, f"numerics.{geometry}.{key}", strict=key != "epsabs",
                        integer=key not in {"epsrel", "epsabs", "lmax_factor"})
        if geometry == "cylinder" and numeric["Nx"] is not None and numeric["Nx"] % 2:
            raise ValueError("numerics.cylinder.Nx must be even")
        order = numeric.get("frequency_order", numeric.get("psd_order"))
        if numeric.get("fs", "psd") == "psd" and order is not None and order < 2:
            raise ValueError(f"numerics.{geometry}: PSD order must be at least 2")
    analysis = settings["analysis"]
    for key in ("refine_extrema", "grid_convergence"):
        if type(analysis[key]) is not bool:
            raise ValueError(f"analysis.{key} must be true or false")
    _number(analysis["max_refine_evaluations"], "analysis.max_refine_evaluations", minimum=3, integer=True)


def size_grid(spec: dict[str, Any]) -> list[float]:
    """Return an endpoint-inclusive linear or logarithmic size grid in nm.

    ``points`` counts every log-grid size, including a separately prepended zero
    when ``minimum_nm`` is zero; ``positive_minimum_nm`` is required in that case.
    Log grids with a positive lower endpoint use ``minimum_nm`` directly.
    """
    if not isinstance(spec, dict):
        raise ValueError("Size scan specification must be an object")
    spacing = spec.get("spacing", "linear")
    if spacing not in ("linear", "log"):
        raise ValueError("Size scan spacing must be 'linear' or 'log'")
    required = {"minimum_nm", "maximum_nm", "step_nm"} if spacing == "linear" else {"minimum_nm", "maximum_nm", "spacing", "points"}
    allowed = required | ({"spacing"} if spacing == "linear" else {"positive_minimum_nm"})
    if not required.issubset(spec) or not spec.keys() <= allowed:
        raise ValueError(f"Invalid {spacing} size scan keys; required: {', '.join(sorted(required))}; allowed: {', '.join(sorted(allowed))}")
    lo = _number(spec["minimum_nm"], "size.minimum_nm")
    hi = _number(spec["maximum_nm"], "size.maximum_nm", strict=True)
    if lo >= hi:
        raise ValueError("Size scan requires 0 <= minimum_nm < maximum_nm")

    if spacing == "log":
        count = int(_number(spec["points"], "size.points", minimum=3 if lo == 0 else 2, integer=True))
        if count > 100001:
            raise ValueError("Too many size points (limit 100001); reduce points")
        if lo == 0:
            if "positive_minimum_nm" not in spec:
                raise ValueError("A log scan starting at zero requires positive_minimum_nm")
            positive_lo = _number(spec["positive_minimum_nm"], "size.positive_minimum_nm", strict=True)
        else:
            if "positive_minimum_nm" in spec:
                raise ValueError("positive_minimum_nm is only valid when minimum_nm is zero")
            positive_lo = lo
        if positive_lo >= hi:
            raise ValueError("Log scan requires positive_minimum_nm < maximum_nm")
        positive_count = count - int(lo == 0)
        start = math.log(positive_lo)
        log_step = (math.log(hi) - start) / (positive_count - 1)
        positive_values = [positive_lo] + [math.exp(start + i * log_step) for i in range(1, positive_count - 1)] + [hi]
        values = ([0.0] if lo == 0 else []) + positive_values
    else:
        step = _number(spec["step_nm"], "size.step_nm", strict=True)
        intervals = (hi - lo) / step
        if not math.isfinite(intervals) or intervals > 100000:
            raise ValueError("Too many size points (limit 100001); increase step_nm")
        count = math.floor(intervals + 1e-12)
        values = [round(lo + i * step, 10) for i in range(count + 1)]
        values[0] = lo
        if len(values) == 1 or not math.isclose(values[-1], hi, rel_tol=0.0, abs_tol=1e-9):
            values.append(hi)
        else:
            values[-1] = hi
    if len(values) > 100001:
        raise ValueError("Too many size points (limit 100001)")
    if any(not math.isfinite(value) or not lo <= value <= hi for value in values):
        raise ValueError("Size grid must contain finite points within the requested range")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("Size grid cannot represent distinct, strictly increasing size points")
    return values


def smoke_settings(settings: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(settings)
    for spec in result["scans"].values():
        if spec.get("spacing", "linear") == "log":
            spec["points"] = 3
        else:
            spec["step_nm"] = (spec["maximum_nm"] - spec["minimum_nm"]) / 2.0
    result["separation"] = {"minimum_nm": 200.0, "maximum_nm": 2000.0, "points": 9, "max_extensions": 1, "extension_factor": 2.0}
    result["analysis"] = {"refine_extrema": False, "max_refine_evaluations": 6, "grid_convergence": False}
    result["numerics"]["plane"].update(frequency_order=2, epsrel=1e-4)
    result["numerics"]["cylinder"].update(Nx=6, Ny=6, nmax=30, frequency_order=2, epsrel=1e-3)
    result["numerics"]["sphere"].update(n_radial=4, m_angular=4, psd_order=2, epsrel=1e-3, lmax_factor=3.0, min_lmax=4, cores=1)
    return result


def source_identity() -> dict[str, Any]:
    """Portable hashes of actual runtime code/data, including selected backends."""
    from .reference_materials import cylinder_material_source, cylinder_reference_provenance
    from .caspwn_adapter import caspwn_source
    folders = {
        "application": ROOT / "Casimir_calculator",
        "cylinder": cylinder_material_source() / "casimir_cylinder",
        "sphere": caspwn_source() / "caspwn",
    }
    sources: dict[str, str] = {}
    for label, folder in folders.items():
        for path in sorted(folder.rglob("*")):
            rel = path.relative_to(folder)
            if not path.is_file() or any(part in {"__pycache__", "tests", ".pytest_cache"} for part in rel.parts):
                continue
            if path.suffix.lower() in {".pyc", ".pyo", ".nbc", ".nbi"}:
                continue
            sources[f"{label}/{rel.as_posix()}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files": sources, "material_sources": cylinder_reference_provenance()}


def run_identity(settings: dict[str, Any], geometries: list[str], profile: str, sources: dict[str, Any]) -> str:
    return digest({"schema": 1, "settings": settings, "geometries": geometries, "profile": profile, "source_files": sources["files"]})


def _point_key(geometry: str, size_nm: float) -> str:
    value = repr(float(size_nm))
    value = value[:-2] if value.endswith(".0") else value
    return f"{geometry}_{value}nm"


def _calculate(geometry: str, size_nm: float, settings: dict[str, Any]) -> dict[str, Any]:
    from .focused_energy import evaluate_size
    return evaluate_size(geometry, size_nm, settings)


def _load_completed(output_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for key, record in manifest["completed"].items():
        if not isinstance(key, str) or "/" in key or "\\" in key or key in {".", ".."}:
            raise ValueError("Invalid checkpoint key")
        expected = output_dir / "points" / f"{key}.json"
        if record.get("file") != f"points/{key}.json":
            raise ValueError(f"Unexpected checkpoint path for {key}")
        if expected.is_symlink() or not expected.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"Checkpoint must remain within its run directory: {expected}")
        payload = read_json(expected)
        if digest(payload) != record["sha256"] or payload.get("run_identity") != manifest["identity"]:
            raise ValueError(f"Checkpoint hash/identity mismatch: {expected}")
        result[key] = payload
    return result


def execute_scan(settings: dict[str, Any], geometries: list[str], output_dir: Path, *, profile: str = "standard", workers: int = 1, resume: bool = False, sources: dict[str, Any] | None = None, evaluator=None) -> dict[str, Any]:
    validate_settings(settings)
    if not geometries or len(set(geometries)) != len(geometries) or any(g not in GEOMETRIES for g in geometries):
        raise ValueError("Select one or more distinct supported geometries")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    sources = source_identity() if sources is None else sources
    identity = run_identity(settings, geometries, profile, sources)
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "run.json"
    planned = [{"geometry": g, "size_nm": s, "key": _point_key(g, s)} for g in geometries for s in size_grid(settings["scans"][g])]
    if resume:
        manifest = read_json(manifest_path)
        if manifest.get("identity") != identity or manifest.get("planned") != planned:
            raise ValueError("Resume refused: settings, selected geometries, profile, or numerical source files changed. Start a new output directory.")
        completed = _load_completed(output_dir, manifest)
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("Output directory is not empty. Use --resume for the same run or choose a new directory.")
        manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
                    "geometries": geometries, "profile": profile, "identity": identity, "sources": sources,
                    "planned": planned, "completed": {}, "errors": {}, "status": "running"}
        completed = {}
    planned_keys = {item["key"] for item in planned}
    if not set(completed).issubset(planned_keys):
        raise ValueError("Checkpoint contains unplanned parameter points")
    pending = [item for item in planned if item["key"] not in completed]
    manifest["status"] = "running"
    manifest["errors"] = {}
    write_json(manifest_path, manifest)

    def accept(item: dict[str, Any], payload: dict[str, Any]) -> None:
        key = item["key"]
        payload = dict(payload, run_identity=identity)
        if payload["geometry"] != item["geometry"] or payload["size_nm"] != item["size_nm"]:
            raise ValueError(f"Backend returned wrong geometry/size for {key}")
        failures = [s["status"] for s in payload["scenarios"] if s["status"] in {"numerical_failure", "unresolved_configuration", "invalid_geometry"}]
        if failures:
            manifest["errors"][key] = {"statuses": failures, "result": payload}
        else:
            write_json(output_dir / "points" / f"{key}.json", payload)
            manifest["completed"][key] = {"file": f"points/{key}.json", "sha256": digest(payload)}
            completed[key] = payload
        write_json(manifest_path, manifest)
        print(f"{len(completed)}/{len(planned)}: {key} — " + ", ".join(s["status"] for s in payload["scenarios"]), flush=True)

    try:
        if workers == 1 or evaluator is not None:
            calculate = evaluator or _calculate
            for item in pending:
                accept(item, calculate(item["geometry"], item["size_nm"], settings))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_calculate, item["geometry"], item["size_nm"], settings): item for item in pending}
                for future in as_completed(futures):
                    accept(futures[future], future.result())
    except BaseException:
        manifest["status"] = "interrupted"
        write_json(manifest_path, manifest)
        raise
    manifest["status"] = "complete" if len(completed) == len(planned) else "partial"
    write_json(manifest_path, manifest)
    from .focused_plots import export_results
    exported = export_results(list(completed.values()), output_dir, settings)
    manifest["artifacts"] = [path.relative_to(output_dir).as_posix() for path in exported]
    write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三种重力下的金板厚度与圆柱/球体半径扫描，输出平衡距离与势阱深度。")
    parser.add_argument("--config", type=Path, help="Scan settings file; default: configs/scan_settings.json")
    parser.add_argument("--geometry", nargs="+", choices=GEOMETRIES, help="选择几何；默认全部")
    parser.add_argument("--workers", type=int, default=1, help="并行尺寸点数，默认1")
    parser.add_argument("--cylinder-method", choices=("pfa", "exact"), help="圆柱算法；默认由设置文件指定，PFA 为快速近似")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path, help="恢复已有扫描目录")
    mode.add_argument("--plot-only", type=Path, help="只重绘已有扫描目录")
    parser.add_argument("--output", type=Path, help="新扫描输出目录，必须为空")
    parser.add_argument("--smoke", action="store_true", help="每种几何3个尺寸点的低精度流程检查")
    parser.add_argument("--check-only", action="store_true", help="核对配置和材料来源，不计算")
    args = parser.parse_args(argv)
    try:
        if args.output and (args.resume or args.plot_only):
            raise ValueError("--output cannot be combined with --resume/--plot-only")
        if args.cylinder_method and (args.resume or args.plot_only):
            raise ValueError("Choose --cylinder-method only for a new scan; saved runs retain their method.")
        if args.plot_only:
            folder = args.plot_only.resolve()
            manifest = read_json(folder / "run.json")
            completed = _load_completed(folder, manifest)
            from .focused_plots import export_results
            export_results(list(completed.values()), folder, manifest["settings"])
            print(f"图像已更新：{folder}")
            return 0
        if args.resume:
            saved = read_json(args.resume / "run.json")
            settings = saved["settings"]
            profile = saved["profile"]
            geometries = args.geometry or saved["geometries"]
            if args.smoke and profile != "smoke":
                raise ValueError("Cannot change a standard run to smoke during resume")
            if args.config is not None:
                requested = read_json(args.config)
                validate_settings(requested)
                requested = smoke_settings(requested) if profile == "smoke" else requested
                if requested != settings:
                    raise ValueError("Resume configuration differs from the saved run. Start a new output directory.")
        else:
            settings = read_json(args.config or ROOT / "configs" / "scan_settings.json")
            if args.cylinder_method:
                settings["numerics"]["cylinder"]["backend"] = args.cylinder_method
            validate_settings(settings)
            profile = "smoke" if args.smoke else "standard"
            settings = smoke_settings(settings) if args.smoke else settings
            geometries = args.geometry or list(GEOMETRIES)
        validate_settings(settings)
        sources = source_identity()
        for geometry in geometries:
            grid = size_grid(settings["scans"][geometry])
            print(f"{geometry}: {grid[0]:g}–{grid[-1]:g} nm，{len(grid)} 个尺寸点，3 种重力", flush=True)
            if geometry == "cylinder":
                backend = settings["numerics"]["cylinder"].get("backend", "exact")
                print(f"圆柱算法：{backend}", flush=True)
                if backend == "pfa":
                    print("老师 PFA：自动 PSD (epsrel=1e-8)，复用 √R 积分；需间距远小于半径。Nx/Ny/nmax/fs/frequency_order/epsrel 为保留的精确法设置。", flush=True)
        if args.check_only:
            print("配置与材料来源检查通过；未执行计算。")
            return 0
        output = args.resume or args.output or ROOT / "outputs" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + profile)
        manifest = execute_scan(settings, geometries, output, profile=profile, workers=args.workers, resume=bool(args.resume), sources=sources)
        print(f"结果目录：{output.resolve()}；状态：{manifest['status']}")
        return 0 if manifest["status"] == "complete" else 1
    except (ValueError, OSError, KeyError) as exc:
        print(f"无法运行：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
