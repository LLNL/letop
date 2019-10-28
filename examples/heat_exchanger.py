from firedrake import (Mesh, FunctionSpace, Function, Constant,
                        SpatialCoordinate)

from ufl import (sin, pi)


from parameters_heat_exch import height, width, inlet_width, dist_center, inlet_depth, shift_center, line_sep

def main():

    mesh = Mesh('./mesh_heat_exchanger.msh')
    x, y = SpatialCoordinate(mesh)
    phi_expr = -y + line_sep \
                        + (y > line_sep + 0.2)*(-2.0*sin((y-line_sep - 0.2)*pi/0.5))*sin(x*pi/width) \
                        - (y < line_sep)*(0.5*sin((y + line_sep/3.0)*pi/(2.0*line_sep/3.0)))* \
                                sin(x*pi*2.0/width)


if __name__ == '__main__':
    main()
