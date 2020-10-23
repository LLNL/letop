import pygmsh as pg
from params import height, width, height_spring, width_spring, height_load, width_load, height_dirch
from params import SPRING, LOAD, DIRCH, ROLL

def main():
    size = 0.01;
    geom = pg.opencascade.Geometry(
            characteristic_length_min=size, characteristic_length_max=size)



    p0 = geom.add_point([0.0, 0.0, 0], size)
    p1 = geom.add_point([width, 0.0, 0], size)
    p2 = geom.add_point([width, height, 0], size)
    p3 = geom.add_point([0.0, height, 0], size)
    p4 = geom.add_point([0.0, height - height_dirch, 0], size)
    p5 = geom.add_point([0.0, height_dirch, 0], size)
    l0 = geom.add_line(p0, p1)
    l1 = geom.add_line(p1, p2)
    l2 = geom.add_line(p2, p3)
    l3 = geom.add_line(p3, p4)
    l4 = geom.add_line(p4, p5)
    l5 = geom.add_line(p5, p0)
    ll0 = geom.add_line_loop([l0, l1, l2, l3, l4, l5])
    main_rect = geom.add_plane_surface(ll0)

    spring_box = geom.add_rectangle([width - width_spring, height / 2.0 - height_spring / 2.0, 0.0], width_spring, height_spring)
    load_box = geom.add_rectangle([0.0, height / 2.0 - height_load / 2.0, 0.0], width_load, height_load)

    geom.add_physical(spring_box, 1)
    geom.add_physical(load_box, 2)

    heat_exchanger = geom.boolean_fragments([main_rect], [spring_box, load_box])

    geom.add_raw_code("""vb1[] = Boundary{{Surface{{ {0} }};}};
                        vb2[] = Boundary{{Surface{{ {1} }};}};
                        vb3[] = Boundary{{Surface{{ {2} }};}};
                        """
                        .format(spring_box.id,
                                load_box.id,
                                heat_exchanger.id
                        ))
    geom.add_raw_code("""Physical Curve({0}) = {{vb1[1]}};""".format(SPRING))
    geom.add_raw_code("""Physical Curve({0}) = {{vb2[3]}};""".format(LOAD))
    geom.add_raw_code("""Physical Curve({0}) = {{vb3[8], vb3[16]}};""".format(DIRCH))
    print(f"l2.id: {l2.id}")
    geom.add_raw_code("""Physical Surface(0) = {bo1[2]};""")


    mesh = pg.generate_mesh(geom, geo_filename="2D_mesh.geo")
    import meshio
    meshio.write("2D_mesh_heat_exchanger.vtk", mesh)

if __name__ == '__main__':
    main()
