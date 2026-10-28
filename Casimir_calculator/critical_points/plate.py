"""Exact layered-plane pressure and analytic gap derivative."""


class PlateForce:
    native_gradient = True
    force_unit = "Pa"

    def __init__(self, settings):
        self.settings = settings

    def __call__(self, t, s, d):
        from ..integrator_finite_temp import SweepConfig, build_default_system
        p = self.settings["physics"]
        config = SweepConfig(teflon_thickness_nm=t, top_plate_thickness_nm=s,
                             temperature_k=p["temperature_K"], material_set=p["material_set"],
                             coating_model=p["coating_model"], **self.settings["profiles"]["plane"])
        system = build_default_system(d * 1e-9, config)
        return float(system.pressure()), float(system.pressuregradient()) * 1e-9
