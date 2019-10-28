import pygmsh as pg
from parameters_heat_exch import height, width, dist_center, inlet_width, inlet_depth, line_sep
from parameters_heat_exch import MOUTH1, MOUTH2, INLET1, INLET2, OUTLET1, OUTLET2, WALLS

assert dist_center > height / 100, "There has to be some separation between the inlets, make dist_center bigger"
assert dist_center + width > line_sep + height / 100, "There has to be some separation between the inlets, make dist_center bigger"
assert height > 0, "height cannot be 0"
assert width > 0, "width cannot be 0"
#geom = pg.built_in.Geometry()
size = 2.e-2;
geom = pg.opencascade.Geometry(
        characteristic_length_min=size, characteristic_length_max=size)

ymax1 = line_sep - (dist_center + inlet_width)
ymax2 = line_sep + dist_center

main_rect = geom.add_rectangle([0.0, 0.0, 0.0], width, height)
mouth_inlet1 = geom.add_rectangle([-inlet_depth, ymax1, 0.0], inlet_depth, inlet_width)
mouth_inlet2 = geom.add_rectangle([-inlet_depth, ymax2, 0.0], inlet_depth, inlet_width)

mouth_outlet1 = geom.add_rectangle([width, ymax1, 0.0], inlet_depth, inlet_width)
mouth_outlet2 = geom.add_rectangle([width, ymax2, 0.0], inlet_depth, inlet_width)

geom.add_physical_surface(mouth_inlet1, MOUTH1)
geom.add_physical_surface(mouth_inlet2, MOUTH2)

heat_exchanger = geom.boolean_union([main_rect, mouth_inlet1, mouth_inlet2, mouth_outlet1, mouth_outlet2])

geom.add_physical_surface(heat_exchanger, 0)

mesh = pg.generate_mesh(geom, geo_filename="mesh_heat_exchanger.geo")

import meshio
meshio.write("mesh_heat_exchanger.msh", mesh)
