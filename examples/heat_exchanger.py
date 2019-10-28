from firedrake import (Mesh, FunctionSpace, Function, Constant,
                        SpatialCoordinate, VectorElement, FiniteElement,
                        TestFunction, interpolate, File, DirichletBC, solve)

from ufl import (sin, pi, split, inner, grad, div, exp, dx, as_vector, derivative)


from parameters_heat_exch import height, width, inlet_width, dist_center, inlet_depth, shift_center, line_sep
from parameters_heat_exch import MOUTH1, MOUTH2, INLET1, INLET2, OUTLET1, OUTLET2, WALLS

def main():

    mesh = Mesh('./mesh_heat_exchanger.msh')
    x, y = SpatialCoordinate(mesh)

    phi_expr = -y + line_sep \
                        + (y > line_sep + 0.2)*(-2.0*sin((y-line_sep - 0.2)*pi/0.5))*sin(x*pi/width) \
                        - (y < line_sep)*(0.5*sin((y + line_sep/3.0)*pi/(2.0*line_sep/3.0)))* \
                                sin(x*pi*2.0/width)

    PHI = FunctionSpace(mesh, 'CG', 1)
    phi = interpolate(phi_expr , PHI)
    File("phi_initial.pvd").write(phi)

    # Build function space
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2*P1
    W = FunctionSpace(mesh, TH)

    U = Function(W)
    u, p = split(U)
    U1, U2 = Function(W), Function(W)
    V = TestFunction(W)
    v, q = split(V)

    f = Constant((0.0, 0.0))
    mu = Constant(1e-2)                   # viscosity
    a_fluid = mu*inner(grad(u), grad(v)) + inner(grad(p), v) + q*div(u)
    darcy_term = inner(u, v)

    alphamax = 1e4
    alphamin = 1e-9
    epsilon = Constant(100.0)

    def hs(phi, epsilon):
        return Constant(alphamax)*Constant(1.0) / ( Constant(1.0) + exp(-epsilon*phi)) + Constant(alphamin)

    def e1f(phi):
        return a_fluid*(dx(0) + dx(MOUTH1) + dx(MOUTH2)) + hs(-phi, epsilon)*darcy_term*dx(0) + alphamax*darcy_term*dx(MOUTH2)
    def e2f(phi):
        return a_fluid*(dx(0) + dx(MOUTH1) + dx(MOUTH2)) + hs(phi, epsilon)*darcy_term*dx(0) + alphamax*darcy_term*dx(MOUTH1)

    u_inflow = 2e-1
    inflow1 = as_vector([u_inflow*sin(((y - (line_sep - (dist_center + inlet_width))) * pi )/ inlet_width), 0.0])
    inflow2 = as_vector([u_inflow*sin(((y - (line_sep + dist_center)) * pi )/ inlet_width), 0.0])

    pressure_outlet = Constant(0.0)
    noslip = Constant((0.0, 0.0))

    # Stokes 1
    bcs1_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs1_2 = DirichletBC(W.sub(0), inflow1, INLET1)
    bcs1_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET1)
    bcs1_4 = DirichletBC(W.sub(0), noslip, INLET2)
    bcs1_5 = DirichletBC(W.sub(0), noslip, OUTLET2)

    # Stokes 2
    bcs2_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs2_2 = DirichletBC(W.sub(0), inflow2, INLET2)
    bcs2_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET2)
    bcs2_4 = DirichletBC(W.sub(0), noslip, INLET1)
    bcs2_5 = DirichletBC(W.sub(0), noslip, OUTLET1)

    # Stokes 1 problem
    parameters = {
            "mat_type" : "aij",
            "ksp_type" : "preonly",
            "pc_type" : "lu",
            "pc_factor_mat_solver_type" : "mumps"
            }

    solve(e1f(phi)==0, U, bcs=[bcs1_1,bcs1_2,bcs1_3,bcs1_4, bcs1_5], solver_parameters=parameters)
    U1.assign(U)
    solve(e2f(phi)==0, U, bcs=[bcs2_1,bcs2_2,bcs2_3,bcs2_4, bcs2_5], solver_parameters=parameters)
    U2.assign(U)

    u, p = U1.split()
    u.rename("Velocity")
    p.rename("Pressure")
    File("u1.pvd").write(u, p)
    u, p = U2.split()
    u.rename("Velocity")
    p.rename("Pressure")
    File("u2.pvd").write(u, p)

if __name__ == '__main__':
    main()
