"""Exact cylinder force per length, retaining separate zero/nonzero profiles.

Calls the unchanged bundled multilayer scattering implementation. The
reflection wave number is the vacuum value, while propagation uses the medium
wave number, as required by the upstream interface.
"""
import importlib
import math

import numpy as np
from scipy.constants import c


class CylinderForce:
    native_gradient = True
    force_unit = "N/m"

    def __init__(self, settings):
        self.settings = settings
        from ..caspwn_adapter import make_materials
        p = settings["physics"]
        self.materials = make_materials(p["material_set"], p["coating_model"])

    def __call__(self, t, s, d):
        from ..cylinder_adapter import LayeredCylinderPlaneSystem
        gold, medium, coating = self.materials
        profile = self.settings["profiles"]["cylinder"]
        temperature = self.settings["physics"]["temperature_K"]
        gap, radius = d * 1e-9, s * 1e-9
        obj = LayeredCylinderPlaneSystem(temperature, gap, radius, gold,
                                        [coating, gold], [t * 1e-9, None], medium)
        nx, ny, order, nmax = profile["zero"]
        obj.calculate("forcegradient", ht_limit=True, Nx=nx, Ny=ny,
                      N=order, nmax=nmax, epsrel=profile["epsrel"])
        zero = np.asarray(obj.n0, dtype=float)
        compute = importlib.import_module("casimir_cylinder.compute_cypl")
        mats, layers = compute.normalize_plane_stack([coating, gold], [t * 1e-9, None])
        reflection = compute.def_reflection_coeff(medium, mats, layers)
        nx, ny, order, nmax = profile["nonzero"]

        def contribution(k):
            eps = medium.epsilon(k * c)
            return compute.Kcontribution(k * math.sqrt(eps), radius, gap,
                                         gold.epsilon(k * c) / eps, nx, obj.x_quad,
                                         obj.y_quad, ny, nmax, reflection,
                                         "forcegradient", reflection_K=k)

        vector = zero + np.asarray(compute.psd_sum(temperature, gap, contribution,
                                                   epsrel=profile["epsrel"], order=order))
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ArithmeticError("Invalid cylinder scattering vector")
        return float(vector[1]), float(vector[2]) * 1e-9
