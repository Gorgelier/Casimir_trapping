from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .optics import layered_reflection

PermittivityFunction = Callable[[float | np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LayeredPlane:
    """A planar stack seen from the intervening medium.

    The first material faces the medium. The last material is the backing
    half-space unless it is deliberately used as a surrounding medium after a
    finite layer.
    """

    materials: Sequence[PermittivityFunction]
    thicknesses_m: Sequence[float] = ()

    def __post_init__(self) -> None:
        materials = tuple(self.materials)
        thicknesses_m = tuple(float(thickness) for thickness in self.thicknesses_m)

        if len(materials) < 1:
            raise ValueError("At least one plane material is required.")
        if len(thicknesses_m) != len(materials) - 1:
            raise ValueError("There must be exactly one thickness per finite layer.")
        if any(not np.isfinite(thickness) or thickness < 0.0 for thickness in thicknesses_m):
            raise ValueError("Layer thicknesses must be finite and non-negative.")
        if not all(callable(material) for material in materials):
            raise ValueError("Each material must be a callable permittivity function.")

        # A zero-thickness finite layer has no optical effect.  Removing it
        # before evaluating the reflection is more than an optimization: at
        # xi=0 a Drude metal can otherwise turn the layer recurrence into an
        # indeterminate 0/0 even though the physical limiting stack is regular.
        retained_materials = [
            material
            for material, thickness in zip(materials[:-1], thicknesses_m)
            if thickness > 0.0
        ]
        retained_materials.append(materials[-1])
        retained_thicknesses = tuple(
            thickness for thickness in thicknesses_m if thickness > 0.0
        )

        object.__setattr__(self, "materials", tuple(retained_materials))
        object.__setattr__(self, "thicknesses_m", retained_thicknesses)

    def reflection(
            self,
            medium_permittivity: PermittivityFunction,
            xi: float | np.ndarray,
            k_parallel: float | np.ndarray,
            polarization: str,
    ) -> np.ndarray:
        epsilons = [medium_permittivity(xi)]
        epsilons.extend(material(xi) for material in self.materials)
        return layered_reflection(
            epsilons,
            self.thicknesses_m,
            xi,
            k_parallel,
            polarization,
        )


def make_plane(
        materials: PermittivityFunction | Sequence[PermittivityFunction],
        thicknesses_m: Sequence[float] = (),
) -> LayeredPlane:
    if callable(materials):
        return LayeredPlane((materials,), thicknesses_m)
    return LayeredPlane(tuple(materials), thicknesses_m)


__all__ = ["LayeredPlane", "PermittivityFunction", "make_plane"]
