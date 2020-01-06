import pygmsh as pg
from params3D import height, depth, width, dist_center, inlet_width, inlet_depth, line_sep, ymin1, ymin2
from params3D import INMOUTH1, INMOUTH2, OUTMOUTH1, OUTMOUTH2, INLET1, INLET2, OUTLET1, OUTLET2, WALLS, DOMAIN

def main():
    #geom = pg.built_in.Geometry()
    size = 0.08;
    geom = pg.opencascade.Geometry(
            characteristic_length_min=size, characteristic_length_max=size)


    main_rect = geom.add_box([0.0, 0.0, 0.0], [width, height, depth])
    mouth_inlet1 = geom.add_box([-inlet_depth, ymin1, 0.0], [inlet_depth, inlet_width, inlet_width])
    mouth_inlet2 = geom.add_box([-inlet_depth, ymin2, 0.0], [inlet_depth, inlet_width, inlet_width])

    mouth_outlet1 = geom.add_box([width, ymin1, 0.0], [inlet_depth, inlet_width, inlet_width])
    mouth_outlet2 = geom.add_box([width, ymin2, 0.0], [inlet_depth, inlet_width, inlet_width])

    print("ymin1 :{}".format(ymin1))
    print("ymin2 :{}".format(ymin2))

    geom.add_physical(mouth_inlet1, INMOUTH1)
    geom.add_physical(mouth_inlet2, INMOUTH2)
    geom.add_physical([main_rect, mouth_outlet2, mouth_outlet1], DOMAIN)

    heat_exchanger = geom.boolean_fragments([main_rect], [mouth_inlet1, mouth_inlet2, mouth_outlet1, mouth_outlet2])

    geom.add_raw_code("""vb1[] = Boundary{{Volume{{ {0} }};}};
                        vb2[] = Boundary{{Volume{{ {1} }};}};
                        vb3[] = Boundary{{Volume{{ {2} }};}};
                        vb4[] = Boundary{{Volume{{ {3} }};}};
                        vb0[] = Boundary{{Volume{{ {4} }};}};"""
                        .format(mouth_inlet1.id,
                                mouth_inlet2.id,
                                mouth_outlet1.id,
                                mouth_outlet2.id,
                                main_rect.id
                        ))
    geom.add_raw_code("""Physical Surface({0}) = {{vb0[],
        		vb1[2], vb1[3], vb1[4], vb1[5],
        		vb2[2], vb2[3], vb2[4], vb2[5],
        		vb3[2], vb3[3], vb3[4], vb3[5],
        		vb4[2], vb4[3], vb4[4], vb4[5]}};"""
                        .format(WALLS)
                        )

    geom.add_raw_code("Physical Surface({0}) -= {{vb1[1], vb2[1], vb3[0], vb4[0]}};\n \
                        Physical Surface({1}) = {{vb1[0]}};\n \
                        Physical Surface({2}) = {{vb3[1]}};\n \
                        Physical Surface({3}) = {{vb2[0]}};\n \
                        Physical Surface({4}) = {{vb4[1]}};"
                        .format(WALLS, INLET1, OUTLET1, INLET2, OUTLET2))

    mesh = pg.generate_mesh(geom, geo_filename="3D_mesh.geo")
    import meshio
    meshio.write("3D_mesh_heat_exchanger.vtk", mesh)

if __name__ == '__main__':
    main()
