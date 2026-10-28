"""Exact sphere force only: gap derivatives are taken from actual forces.

The historical native force-gradient observable is deliberately not used.
"""


class SphereForce:
    native_gradient = False
    force_unit = "N"

    def __init__(self, settings):
        from ..caspwn_adapter import load_plane_sphere_system, make_materials
        self.settings = settings
        self.factory = load_plane_sphere_system()
        p = settings["physics"]
        self.materials = make_materials(p["material_set"], p["coating_model"])

    def __call__(self, t, s, d):
        gold, medium, coating = self.materials
        system = self.factory(T=self.settings["physics"]["temperature_K"],
                              L=d * 1e-9, R=s * 1e-9, mat_sphere=gold,
                              mat_plane=coating, mat_medium=medium)
        system.add_plane_layers([gold], [t * 1e-9])
        force = float(system.calculate("force", **self.settings["profiles"]["sphere"]))
        return force, None
