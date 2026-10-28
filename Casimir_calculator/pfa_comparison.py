"""Shared, no-gravity PFA kernels for the Au/ethanol/PTFE/Au campaign.

The planar Lifshitz free energy per area is W(d).  We use the leading
Derjaguin conventions E_sphere = 2*pi*R*integral_d^infinity W(z) dz and
E_cylinder/L = integral_-infinity^infinity W(d+x*x/(2*R)) dx.  The latter
is exactly the convention in casimir_cylinder.pfa_cypl, not integration
over the finite projected footprint of a circle.  Both tend to zero at
infinite separation.  Thus zero-gravity equilibrium is independent of R,
and energy scales as R for a sphere or sqrt(R) for a cylinder.

The gap integrals are performed analytically (Li_2 and Li_3/2). Optical
models come directly from this project's teacher-material adapters.
Gauss-Legendre radial quadrature and finite PSD order are explicit and
can be doubled independently.  No scattering result enters this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma, spence, zeta

from .constants import BOLTZMANN, C, HBAR
from .material_sets import (
    DEFAULT_COATING_MODEL, DEFAULT_MATERIAL_SET, select_coating,
    select_gold_and_medium,
)
from .psd import psd
from .reference_materials import cylinder_reference_provenance


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isreal(value):
        raise ValueError(f"{name} must be a positive finite real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number")
    return value


@lru_cache(maxsize=2)
def _endpoint_coefficients(order):
    return np.array([(-1.)**j * zeta(order-j) / math.factorial(j)
                     for j in range(22)])


def _positive_polylog(z, order):
    """Real Li_s(z), 0 <= z < 1, s = 1/2 or 3/2, to roundoff."""
    result = np.empty_like(z)
    small = z <= .5
    q = z[small]
    # At |q| <= 1/2, 60 terms leave less than 2e-18 absolute error.
    term = q.copy()
    total = term.copy()
    for n in range(2, 61):
        term *= q
        total += term / n**order
    result[small] = total
    mu = -np.log(z[~small])
    result[~small] = (gamma(1-order) * mu**(order-1)
                      + np.polynomial.polynomial.polyval(mu, _endpoint_coefficients(order)))
    return result


def real_polylog(z, order):
    """Li_s(z) for -1 < z < 1 and s=1/2, 3/2 or 2."""
    z = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(z)) or np.any(np.abs(z) >= 1):
        raise ValueError("polylog argument must be finite and strictly between -1 and 1")
    if order == 2:
        return spence(1-z)
    if order not in (.5, 1.5):
        raise ValueError("Supported orders are 0.5, 1.5, and 2")
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = _positive_polylog(z[positive], order)
    q = -z[~positive]
    # Li_s(-q) = 2^(1-s) Li_s(q^2) - Li_s(q).
    result[~positive] = 2**(1-order)*_positive_polylog(q*q, order)-_positive_polylog(q, order)
    return result


@lru_cache(maxsize=12)
def _radial_rule(order):
    if isinstance(order, bool) or int(order) != order or order < 4:
        raise ValueError("radial_order must be an integer at least 4")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    # t = kappa_medium*d; integrate from each mode's lower limit a to a+40.
    # Exponentially attenuated tail is O(exp(-80)); the short first panels
    # resolve the logarithmic n=0 endpoint and thin-coating crossover.
    boundaries = [0., .01, .1, 1., 5., 15., 40.]
    values, quadrature_weights = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        values.append(left + (right-left)*(nodes+1)/2)
        quadrature_weights.append((right-left)*weights/2)
    return np.concatenate(values), np.concatenate(quadrature_weights)


@dataclass
class PFAKernel:
    thickness_nm: float
    temperature_k: float = 300.
    frequency_order: int = 128
    radial_order: int = 32
    material_set: str = DEFAULT_MATERIAL_SET
    coating_model: str = DEFAULT_COATING_MODEL

    def __post_init__(self):
        self.thickness_nm = _positive(self.thickness_nm, "thickness_nm")
        self.temperature_k = _positive(self.temperature_k, "temperature_k")
        poles, weights = psd(self.frequency_order)
        self._xi = np.r_[0., poles*BOLTZMANN*self.temperature_k/HBAR]
        self._frequency_weights = np.r_[.5, weights]
        gold, medium = select_gold_and_medium(self.material_set)
        coating = select_coating(self.coating_model)
        self._eps_gold = gold(self._xi)[:, None]
        self._eps_medium = medium(self._xi)[:, None]
        self._eps_coating = coating(self._xi)[:, None]
        self._nodes, self._weights = _radial_rule(self.radial_order)

    def provenance(self):
        return {
            "model": "leading Derjaguin/parabolic-gap PFA",
            "gravity_acceleration_m_s2": 0.,
            "temperature_k": self.temperature_k,
            "thickness_nm": self.thickness_nm,
            "frequency_summation": "PSD", "frequency_order": self.frequency_order,
            "radial_order_per_panel": self.radial_order,
            "radial_panels": [0., .01, .1, 1., 5., 15., 40.],
            "energy_reference": "E(d=infinity)=0",
            "sphere_formula": "E=2*pi*R*integral_d^infinity W(z) dz",
            "cylinder_formula": "E/L=integral_-infinity^infinity W(d+x^2/(2R)) dx",
            "sphere_units": "J; force N",
            "cylinder_units": "J/m; force N/m",
            "material_reference": cylinder_reference_provenance(),
            "validity": "PFA requires gap/radius << 1; numerical convergence does not establish PFA accuracy",
        }

    def _optical_products(self, distance_m):
        k0d = (self._xi*distance_m/C)[:, None]
        a2 = self._eps_medium*k0d*k0d
        t = np.sqrt(a2) + self._nodes[None, :]
        p2 = t*t-a2
        tc = np.sqrt(p2+self._eps_coating*k0d*k0d)
        tg = np.sqrt(p2+self._eps_gold*k0d*k0d)
        # Exact static Drude TM limit; TE=0 follows from identical t=tc=tg.
        rmg_tm = (self._eps_gold*t-self._eps_medium*tg)/(self._eps_gold*t+self._eps_medium*tg)
        rmc_tm = (self._eps_coating*t-self._eps_medium*tc)/(self._eps_coating*t+self._eps_medium*tc)
        rcg_tm = (self._eps_gold*tc-self._eps_coating*tg)/(self._eps_gold*tc+self._eps_coating*tg)
        rmg_tm[0] = 1.
        rcg_tm[0] = 1.
        rmg_te = (t-tg)/(t+tg)
        rmc_te = (t-tc)/(t+tc)
        rcg_te = (tc-tg)/(tc+tg)
        layer_decay = np.exp(-2*tc*self.thickness_nm*1.e-9/distance_m)
        rplane_tm = (rmc_tm+rcg_tm*layer_decay)/(1+rmc_tm*rcg_tm*layer_decay)
        rplane_te = (rmc_te+rcg_te*layer_decay)/(1+rmc_te*rcg_te*layer_decay)
        propagation = np.exp(-2*t)
        return t, rmg_tm*rplane_tm*propagation, rmg_te*rplane_te*propagation

    def evaluate(self, distance_nm, geometry, radius_nm=1000.):
        """Return no-gravity energy/force in J,N (sphere) or J/m,N/m (cylinder)."""
        d = _positive(distance_nm, "distance_nm")*1.e-9
        radius = _positive(radius_nm, "radius_nm")*1.e-9
        if geometry not in {"sphere", "cylinder"}:
            raise ValueError("geometry must be sphere or cylinder")
        t, qtm, qte = self._optical_products(d)
        w = self._frequency_weights[:, None]*self._weights[None, :]
        kt = BOLTZMANN*self.temperature_k
        if geometry == "sphere":
            energy = -kt*radius/(2*d)*np.sum(w*(real_polylog(qtm, 2)+real_polylog(qte, 2)))
            force = kt*radius/d**2*np.sum(w*t*(np.log1p(-qtm)+np.log1p(-qte)))
        else:
            energy = -kt*np.sqrt(radius)/(2*np.sqrt(np.pi)*d**1.5)*np.sum(
                w*np.sqrt(t)*(real_polylog(qtm, 1.5)+real_polylog(qte, 1.5)))
            force = -kt*np.sqrt(radius)/(np.sqrt(np.pi)*d**2.5)*np.sum(
                w*t**1.5*(real_polylog(qtm, .5)+real_polylog(qte, .5)))
        return {"energy": float(energy), "force": float(force)}

    def equilibrium(self, geometry, bracket_nm=(1., 1000.), radius_nm=1000.):
        """Find a stable force root and report escape-to-infinity depth -E_min.

        The supplied bracket must contain a positive-to-negative force crossing.
        Callers should inspect the energy curve for competing interior barriers.
        """
        left, right = (float(v) for v in bracket_nm)
        f = lambda d: self.evaluate(d, geometry, radius_nm)["force"]
        if not (f(left) > 0 and f(right) < 0):
            raise ValueError("bracket must contain a stable positive-to-negative force crossing")
        d = brentq(f, left, right, xtol=1.e-8, rtol=1.e-12)
        result = self.evaluate(d, geometry, radius_nm)
        return {
            "geometry": geometry, "thickness_nm": self.thickness_nm,
            "radius_nm": float(radius_nm), "equilibrium_distance_nm": float(d),
            "equilibrium_energy_native": result["energy"],
            "trap_depth_native": -result["energy"],
            "trap_depth_reporting": -result["energy"]/(BOLTZMANN*self.temperature_k)*(
                1.e-6 if geometry == "cylinder" else 1.),
            "force_at_equilibrium_native": result["force"],
            "energy_unit": "J/m" if geometry == "cylinder" else "J",
            "reporting_unit": "k_B T / um" if geometry == "cylinder" else "k_B T",
            "trap_depth_definition": "escape to infinity: 0 - E_min",
            "gravity_case": "gravity_zero",
        }


__all__ = ["PFAKernel", "real_polylog"]
