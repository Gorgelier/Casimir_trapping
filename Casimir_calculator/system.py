from __future__ import annotations

from dataclasses import dataclass

from .finite_temperature import FiniteTemperatureConfig, calculate_per_area
from .plane import LayeredPlane, PermittivityFunction, make_plane


@dataclass(frozen=True)
class CasimirSystem:
    """Plane-plane Lifshitz system.

    The public shape intentionally follows the `system` classes in califorcia
    and casimir_cylinder: build a system, then call `calculate(observable)`.
    """

    temperature_k: float
    distance_m: float
    left_plane: LayeredPlane
    right_plane: LayeredPlane
    medium_permittivity: PermittivityFunction
    fs: str = "psd"
    epsrel: float = 1.0e-8
    epsabs: float = 0.0
    frequency_order: int | None = None
    xi_nodes: int | None = None
    k_nodes: int | None = None

    def __post_init__(self) -> None:
        if self.distance_m <= 0.0:
            raise ValueError("distance_m must be positive.")
        if not callable(self.medium_permittivity):
            raise ValueError("medium_permittivity must be callable.")

    @property
    def integration_config(self) -> FiniteTemperatureConfig:
        return FiniteTemperatureConfig(
            temperature_k=self.temperature_k,
            frequency_order=self.frequency_order,
            fs=self.fs,
            epsrel=self.epsrel,
            epsabs=self.epsabs,
            xi_nodes=self.xi_nodes,
            k_nodes=self.k_nodes,
        )

    def calculate(
        self,
        observable: str,
        ht_limit: bool = False,
        fs: str | None = None,
        epsrel: float | None = None,
        epsabs: float | None = None,
        N: int | None = None,
    ) -> float:
        config = FiniteTemperatureConfig(
            temperature_k=self.temperature_k,
            frequency_order=self.frequency_order if N is None else N,
            fs=self.fs if fs is None else fs,
            epsrel=self.epsrel if epsrel is None else epsrel,
            epsabs=self.epsabs if epsabs is None else epsabs,
            # A per-call N is the authoritative override.  Passing the legacy
            # alias as well would either overwrite it or create a false
            # conflict with the system-level setting.
            xi_nodes=self.xi_nodes if N is None else None,
            k_nodes=self.k_nodes,
        )
        return calculate_per_area(
            observable,
            self.distance_m,
            self.left_plane,
            self.right_plane,
            self.medium_permittivity,
            config,
            ht_limit=ht_limit,
        )

    def energy(self, **kwargs) -> float:
        return self.calculate("energy", **kwargs)

    def pressure(self, **kwargs) -> float:
        return self.calculate("pressure", **kwargs)

    def pressuregradient(self, **kwargs) -> float:
        return self.calculate("pressuregradient", **kwargs)


def system(
    temperature_k: float,
    distance_m: float,
    left_materials,
    right_materials,
    medium_permittivity: PermittivityFunction,
    deltaL=None,
    deltaR=None,
    *,
    delta_left_m=None,
    delta_right_m=None,
    fs: str = "psd",
    epsrel: float = 1.0e-8,
    epsabs: float = 0.0,
    N: int | None = None,
    xi_nodes: int | None = None,
    k_nodes: int | None = None,
) -> CasimirSystem:
    if deltaL is not None and delta_left_m is not None:
        raise ValueError("Use either deltaL or delta_left_m, not both.")
    if deltaR is not None and delta_right_m is not None:
        raise ValueError("Use either deltaR or delta_right_m, not both.")

    left_thicknesses = delta_left_m if delta_left_m is not None else deltaL
    right_thicknesses = delta_right_m if delta_right_m is not None else deltaR

    return CasimirSystem(
        temperature_k=temperature_k,
        distance_m=distance_m,
        left_plane=make_plane(left_materials, () if left_thicknesses is None else left_thicknesses),
        right_plane=make_plane(right_materials, () if right_thicknesses is None else right_thicknesses),
        medium_permittivity=medium_permittivity,
        fs=fs,
        epsrel=epsrel,
        epsabs=epsabs,
        frequency_order=N,
        xi_nodes=xi_nodes,
        k_nodes=k_nodes,
    )


__all__ = ["CasimirSystem", "system"]
