from firedrake import (Mesh, SpatialCoordinate, FunctionSpace, interpolate)

from ufl import sin, pi

from parameters_heat_exch import height, width, inlet_width, dist_center, inlet_depth, shift_center, line_sep
from parameters_heat_exch import OUTMOUTH1, OUTMOUTH2, INMOUTH1, INMOUTH2, INLET1, INLET2, OUTLET1, OUTLET2, WALLS

from optimization_problem import OptimizationProblem

mesh = Mesh('./mesh_heat_exchanger.msh')

X = SpatialCoordinate(mesh)
x, y = X[0], X[1]
phi_expr = -y + line_sep \
            + (y > line_sep + 0.2)*(-2.0*sin((y-line_sep - 0.2)*pi/0.5))*sin(x*pi/width) \
            - (y < line_sep)*(0.5*sin((y + line_sep/3.0)*pi/(2.0*line_sep/3.0)))* \
                    sin(x*pi*2.0/width)

opti_problem = OptimizationProblem(mesh)

PHI = FunctionSpace(mesh, 'CG', 1)
phi = interpolate(phi_expr , PHI)
J = opti_problem.cost_function_evaluation(phi)
dJ = opti_problem.derivative_evaluation(phi)
print("Cost function: {0:.5f}".format(J))
import numpy as np
print("Sum for dJ: {0:.5f}".format(np.sum(np.abs(dJ.dat.data))))
