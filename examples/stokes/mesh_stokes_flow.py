import pygmsh as pg

INMOUTH1, INMOUTH2, OUTMOUTH1, OUTMOUTH2 = 1, 2, 3, 4
WALLS = 5
def main():
    size = 1.0e-2;
    geom = pg.built_in.Geometry()

    INLET1_SEP = 0.2
    INLET1_MOUTH = 0.2
    INLETS_DIST = 0.2
    INLET2_SEP = 0.2
    INLET2_MOUTH = 0.2

    WIDTH = 1.0

    OUTLET1_SEP = 0.2
    OUTLET1_MOUTH = 0.2
    OUTLETS_DIST = 0.2
    OUTLET2_SEP = 0.2
    OUTLET2_MOUTH = 0.2


    p1 = geom.add_point([0.0, 0.0, 0.0],size)
    p2 = geom.add_point([0.0, INLET1_SEP, 0.0],size)
    p3 = geom.add_point([0.0, INLET1_SEP + INLET1_MOUTH, 0.0],size)
    p4 = geom.add_point([0.0, INLET1_SEP + INLET1_MOUTH + INLETS_DIST, 0.0],size)
    p5 = geom.add_point([0.0, INLET1_SEP + INLET1_MOUTH + INLETS_DIST + INLET2_MOUTH, 0.0],size)
    p6 = geom.add_point([0.0, INLET1_SEP + INLET1_MOUTH + INLETS_DIST + INLET2_MOUTH + INLET2_SEP, 0.0],size)

    p7 = geom.add_point([WIDTH, OUTLET1_SEP + OUTLET1_MOUTH + OUTLETS_DIST + OUTLET2_MOUTH + OUTLET2_SEP, 0.0],size)
    p8 = geom.add_point([WIDTH, OUTLET1_SEP + OUTLET1_MOUTH + OUTLETS_DIST + OUTLET2_MOUTH, 0.0],size)
    p9 = geom.add_point([WIDTH, OUTLET1_SEP + OUTLET1_MOUTH + OUTLETS_DIST, 0.0],size)
    p10 = geom.add_point([WIDTH, OUTLET1_SEP + OUTLET1_MOUTH, 0.0],size)
    p11 = geom.add_point([WIDTH, OUTLET1_SEP, 0.0],size)
    p12 = geom.add_point([WIDTH, 0.0, 0.0],size)


    l0 = geom.add_line(p1, p2)
    l1 = geom.add_line(p2, p3)
    l2 = geom.add_line(p3, p4)
    l3 = geom.add_line(p4, p5)
    l4 = geom.add_line(p5, p6)
    l5 = geom.add_line(p6, p7)
    l6 = geom.add_line(p7, p8)
    l7 = geom.add_line(p8, p9)
    l8 = geom.add_line(p9, p10)
    l9 = geom.add_line(p10, p11)
    l10 = geom.add_line(p11, p12)
    l11 = geom.add_line(p12, p1)

    ll = geom.add_line_loop([l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11])
    surf = geom.add_plane_surface(ll)

    geom.add_physical(l1, INMOUTH1)
    geom.add_physical(l3, INMOUTH2)
    geom.add_physical(l9, OUTMOUTH1)
    geom.add_physical(l7, OUTMOUTH2)
    geom.add_physical([l0, l2, l4, l5, l6, l8, l10, l11], WALLS)
    geom.add_physical(surf, 0)


    mesh = pg.generate_mesh(geom, geo_filename="mesh_stokes.geo")

    import meshio
    meshio.write("mesh_stokes.vtk", mesh)

if __name__ == '__main__':
    main()
