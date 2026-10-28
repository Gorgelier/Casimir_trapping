from __future__ import annotations

HBAR = 1.054571817e-34
C = 299792458.0
E_CHARGE = 1.602176634e-19
EV_TO_RAD_S = E_CHARGE / HBAR
NANOMETER = 1.0e-9
BOLTZMANN = 1.380649e-23
# Campaign-wide buoyancy reference values. Provenance and reference conditions
# are recorded in configs/parameter_sweeps.yaml and docs/PHYSICAL_CONVENTIONS.md.
GOLD_DENSITY = 19320.0  # kg/m^3, pure Au; NIST STAR room-temperature value.
ETHANOL_DENSITY = 785.06  # kg/m^3, anhydrous ethanol at 298.15 K and about 0.1 MPa.
G_ACCEL = 9.81

__all__ = [
    "C",
    "E_CHARGE",
    "EV_TO_RAD_S",
    "HBAR",
    "NANOMETER",
    "BOLTZMANN",
    "GOLD_DENSITY",
    "ETHANOL_DENSITY",
    "G_ACCEL",
]
