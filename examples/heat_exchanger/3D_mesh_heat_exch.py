import pygmsh as pg
from parameters_heat_exch import height, width, dist_center, inlet_width, inlet_depth, line_sep
from parameters_heat_exch import INMOUTH1, INMOUTH2, OUTMOUTH1, OUTMOUTH2, INLET1, INLET2, OUTLET1, OUTLET2, WALLS, DOMAIN

assert dist_center > height / 100, "There has to be some separation between the inlets, make dist_center bigger"
assert dist_center + width > line_sep + height / 100, "There has to be some separation between the inlets, make dist_center bigger"
assert height > 0, "height cannot be 0"
assert width > 0, "width cannot be 0"

ymax1 = line_sep - (dist_center + inlet_width)
ymax2 = line_sep + dist_center

def main():
    #geom = pg.built_in.Geometry()
    size = 4.e-2;
    geom = pg.opencascade.Geometry(
            characteristic_length_min=size, characteristic_length_max=size)


    main_rect = geom.add_box([0.0, 0.0, 0.0], [width, height, height])
    mouth_inlet1 = geom.add_box([-inlet_depth, ymax1, 0.0], [inlet_depth, inlet_width, inlet_width])
    mouth_inlet2 = geom.add_box([-inlet_depth, ymax2, 0.0], [inlet_depth, inlet_width, inlet_width])

    mouth_outlet1 = geom.add_box([width, ymax1, 0.0], [inlet_depth, inlet_width, inlet_width])
    mouth_outlet2 = geom.add_box([width, ymax2, 0.0], [inlet_depth, inlet_width, inlet_width])

    geom.add_physical(mouth_inlet1, INMOUTH1)
    geom.add_physical(mouth_inlet2, INMOUTH2)
    geom.add_physical(mouth_outlet1, OUTMOUTH1)
    geom.add_physical(mouth_outlet2, OUTMOUTH2)
    geom.add_physical([main_rect, mouth_outlet2, mouth_outlet1], DOMAIN)

    heat_exchanger = geom.boolean_union([main_rect, mouth_inlet1, mouth_inlet2, mouth_outlet1, mouth_outlet2])

    mesh = pg.generate_mesh(geom, geo_filename="3D_mesh_heat_exchanger.geo")
    import meshio
    meshio.write("3D_mesh_heat_exchanger.vtk", mesh)

if __name__ == '__main__':
    main()
