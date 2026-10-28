from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .materials import (
    ethanol_cylinder_reference_permittivity,
    ethanol_gudarzi2021_optical_static_permittivity,
    ethanol_gudarzi2021_permittivity,
    ethanol_permittivity,
    gold_califorcia_drude_permittivity,
    gold_cylinder_reference_permittivity,
    gold_gudarzi2021_permittivity,
    gold_lorentz_drude_permittivity,
    ptfe_hd_permittivity,
    ptfe_ld_permittivity,
    ptfe_zwol_permittivity,
    teflon_af1601_permittivity,
    teflon_af2400_permittivity,
    teflon_gudarzi2021_permittivity,
    teflon_cylinder_reference_permittivity,
)

Permittivity: TypeAlias = Callable[[float | np.ndarray], np.ndarray]

DEFAULT_MATERIAL_SET = "cylinder-reference"
DEFAULT_COATING_MODEL = "teflon-cylinder-reference"

COATING_MODELS: dict[str, Permittivity] = {
    DEFAULT_COATING_MODEL: teflon_cylinder_reference_permittivity,
    "af1601": teflon_af1601_permittivity,
    "af1600": teflon_af1601_permittivity,
    "af2400": teflon_af2400_permittivity,
    "ptfe-ld": ptfe_ld_permittivity,
    "ptfe-hd": ptfe_hd_permittivity,
    "ptfe-califorcia": ptfe_zwol_permittivity,
    "teflon-califorcia": ptfe_zwol_permittivity,
    "teflon-gudarzi2021": teflon_gudarzi2021_permittivity,
    "ptfe-gudarzi2021": teflon_gudarzi2021_permittivity,
}

MATERIAL_SETS = {
    DEFAULT_MATERIAL_SET,
    "califorcia",
    "default",
    "gudarzi2021",
    "gudarzi2021-ethanol-optical-static",
}


def select_gold_and_medium(material_set: str) -> tuple[Permittivity, Permittivity]:
    if material_set == DEFAULT_MATERIAL_SET:
        return gold_cylinder_reference_permittivity, ethanol_cylinder_reference_permittivity
    if material_set == "califorcia":
        return gold_califorcia_drude_permittivity, ethanol_permittivity
    if material_set == "gudarzi2021":
        return gold_gudarzi2021_permittivity, ethanol_gudarzi2021_permittivity
    if material_set == "gudarzi2021-ethanol-optical-static":
        return gold_gudarzi2021_permittivity, ethanol_gudarzi2021_optical_static_permittivity
    if material_set == "default":
        return gold_lorentz_drude_permittivity, ethanol_permittivity
    allowed = ", ".join(sorted(MATERIAL_SETS))
    raise ValueError(f"Unsupported material_set '{material_set}'. Allowed: {allowed}")


def select_coating(coating_model: str) -> Permittivity:
    try:
        return COATING_MODELS[coating_model]
    except KeyError as exc:
        allowed = ", ".join(sorted(COATING_MODELS))
        raise ValueError(f"Unsupported coating_model '{coating_model}'. Allowed: {allowed}") from exc


@dataclass(frozen=True)
class ScatteringMaterial:
    """Small material adapter for califorcia, caspwn, and cylinder-style APIs."""

    name: str
    materialclass: str
    permittivity: Permittivity

    def epsilon(self, xi: float) -> float:
        return float(np.asarray(self.permittivity(xi), dtype=float))


def make_scattering_materials(
    material_set: str,
    coating_model: str,
) -> tuple[ScatteringMaterial, ScatteringMaterial, ScatteringMaterial]:
    gold_eps, ethanol_eps = select_gold_and_medium(material_set)
    coating_eps = select_coating(coating_model)
    return (
        ScatteringMaterial(f"gold-{material_set}", "drude", gold_eps),
        ScatteringMaterial(f"ethanol-{material_set}", "dielectric", ethanol_eps),
        ScatteringMaterial(coating_model, "dielectric", coating_eps),
    )


__all__ = [
    "COATING_MODELS",
    "DEFAULT_COATING_MODEL",
    "DEFAULT_MATERIAL_SET",
    "MATERIAL_SETS",
    "Permittivity",
    "ScatteringMaterial",
    "make_scattering_materials",
    "select_coating",
    "select_gold_and_medium",
]
