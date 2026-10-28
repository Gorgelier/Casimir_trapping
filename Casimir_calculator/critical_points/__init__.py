"""Finite-gap critical points of exact-scattering equilibrium branches.

Coordinates are in nm. Native forces are Pa (plane), N/m (cylinder),
and N (sphere). No proximity-force approximation is used here.
"""

from .solver import solve_critical_point

__all__ = ["solve_critical_point"]
