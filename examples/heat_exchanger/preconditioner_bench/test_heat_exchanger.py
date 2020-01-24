from firedrake import *
import numpy as np
from ufl import replace, conditional
from ufl import min_value, max_value

set_log_level(ERROR)


def main(mesh, stokes_parameters, temperature_parameters):
    output_dir = "./"

    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    X = SpatialCoordinate(mesh)
    dim = mesh.geometric_dimension()
    if dim == 2:
        x, y = X[0], X[1]
        from params import (INMOUTH2, INMOUTH1, line_sep, dist_center, inlet_width,
                                        WALLS, INLET1, INLET2, OUTLET1, OUTLET2, width)
    elif dim == 3:
        x, y, z = X[0], X[1], X[2]
        from params3D import (INMOUTH2, INMOUTH1, line_sep, dist_center, inlet_width, ymin1, ymin2,
                                        WALLS, INLET1, INLET2, OUTLET1, OUTLET2, width)

    phi_expr = -y + line_sep \
                + (y > line_sep + 0.2)*(-2.0*sin((y-line_sep - 0.2)*pi/0.5))*sin(x*pi/width) \
                - (y < line_sep)*(0.5*sin((y + line_sep/3.0)*pi/(2.0*line_sep/3.0)))* \
                        sin(x*pi*2.0/width)

    PHI = FunctionSpace(mesh, 'CG', 1)
    phi = interpolate(phi_expr , PHI)
    phi.rename("LevelSet")
    phi_pvd = File(output_dir + "phi_evolution.pvd")
    phi_pvd.write(phi)

    mu = Constant(1.0)                   # viscosity
    alphamax = 2.5 * mu / (1e-5)
    alphamin = 1e-12
    epsilon = Constant(1000.0)
    u_inflow = 2e-1
    tin1 = Constant(10.0)
    tin2 = Constant(100.0)

    P2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2*P1
    W = FunctionSpace(mesh, TH)

    U = TrialFunction(W)
    u, p = split(U)
    V = TestFunction(W)
    v, q = split(V)

    def hs(phi, epsilon):
        return Constant(alphamax)*Constant(1.0) / ( Constant(1.0) + exp(-epsilon*phi)) + Constant(alphamin)

    Q = FunctionSpace(mesh, 'DG', 0)
    File("hs_phi.pvd").write(interpolate(hs(phi_expr, epsilon), Q))
    def stokes(phi, BLOCK_MOUTH):
        a_fluid = (mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u))
        darcy_term = inner(u, v)
        if BLOCK_MOUTH == 2:
            alpha_inlet_other = Constant(alphamax)*conditional(
                            And(lt(x, 0.0),
                                lt(y, line_sep)), 1.0, 1e-8)
            alpha_inlet_mine = hs(phi, epsilon)*conditional(
                            Or(
                            And(lt(x, 0.0),
                                gt(y, line_sep)),
                                And(gt(x, width),
                                    gt(y, line_sep))), 1e-8, 1.0)
        elif BLOCK_MOUTH == 3:
            alpha_inlet_other = Constant(alphamax)*conditional(
                            And(lt(x, 0.0),
                                gt(y, line_sep)), 1.0, 1e-8)
            alpha_inlet_mine = hs(phi, epsilon)*conditional(
                            Or(
                            And(lt(x, 0.0),
                                lt(y, line_sep)),
                                And(gt(x, width),
                                    lt(y, line_sep))), 1e-8, 1.0)


        File("alpha.pvd").write(interpolate(alpha_inlet_other, Q))



        return a_fluid*dx + alpha_inlet_mine*darcy_term*dx + alpha_inlet_other*darcy_term*dx
    def stokesP(phi, BLOCK_MOUTH):
        a_fluid = (mu*inner(grad(u), grad(v)) + p*q)
        darcy_term = inner(u, v)
        if BLOCK_MOUTH == 2:
            alpha_inlet_other = Constant(alphamax)*conditional(
                            And(lt(x, 0.0),
                                lt(y, line_sep)), 1.0, 1e-8)
            alpha_inlet_mine = hs(phi, epsilon)*conditional(
                            Or(
                            And(lt(x, 0.0),
                                gt(y, line_sep)),
                                And(gt(x, width),
                                    gt(y, line_sep))), 1e-8, 1.0)
        elif BLOCK_MOUTH == 3:
            alpha_inlet_other = Constant(alphamax)*conditional(
                            And(lt(x, 0.0),
                                gt(y, line_sep)), 1.0, 1e-8)
            alpha_inlet_mine = hs(phi, epsilon)*conditional(
                            Or(
                            And(lt(x, 0.0),
                                lt(y, line_sep)),
                                And(gt(x, width),
                                    lt(y, line_sep))), 1e-8, 1.0)


        File("alpha.pvd").write(interpolate(alpha_inlet_other, Q))



        return a_fluid*dx + alpha_inlet_mine*darcy_term*dx + alpha_inlet_other*darcy_term*dx


    # Dirichelt boundary conditions
    if dim == 2:
        inflow1 = as_vector([u_inflow*sin(((y - (line_sep - (dist_center + inlet_width))) * pi )/ inlet_width), 0.0])
        inflow2 = as_vector([u_inflow*sin(((y - (line_sep + dist_center)) * pi )/ inlet_width), 0.0])
        noslip = Constant((0.0, 0.0))
    elif dim == 3:
        inflow1 = as_vector([u_inflow*sin(z * pi / inlet_width) * sin((y - ymin1) * pi / inlet_width), 0.0, 0.0])
        inflow2 = as_vector([u_inflow*sin(z * pi / inlet_width) * sin((y - ymin2) * pi / inlet_width), 0.0, 0.0])
        noslip = Constant((0.0, 0.0, 0.0))


    # Stokes 1
    bcs1_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs1_2 = DirichletBC(W.sub(0), inflow1, INLET1)
    bcs1_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET1)
    bcs1_4 = DirichletBC(W.sub(0), noslip, INLET2)
    bcs1_5 = DirichletBC(W.sub(0), noslip, OUTLET2)
    bcs1 = [bcs1_1,bcs1_2,bcs1_3,bcs1_4, bcs1_5]

    # Stokes 2
    bcs2_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs2_2 = DirichletBC(W.sub(0), inflow2, INLET2)
    bcs2_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET2)
    bcs2_4 = DirichletBC(W.sub(0), noslip, INLET1)
    bcs2_5 = DirichletBC(W.sub(0), noslip, OUTLET1)
    bcs2 = [bcs2_1,bcs2_2,bcs2_3,bcs2_4, bcs2_5]

    # Forward problems
    U1, U2 = Function(W), Function(W)
    L = inner(Constant(tuple([0.0 for _ in range(dim+1)])), V)*dx

    multigrid = False
    if multigrid:
        aP = None
    else:
        aP = None

    problem = LinearVariationalProblem(stokes(-phi, INMOUTH2), L, U1, aP=aP, bcs=bcs1)
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
    solver_stokes1 = LinearVariationalSolver(problem, solver_parameters=stokes_parameters) #, nullspace=nullspace)
    solver_stokes1.solve()
    print("Iterations Stokes 1 # {}".format(solver_stokes1.snes.ksp.getIterationNumber()))
    u1, _ = U1.split()
    File("u1.pvd").write(u1)


    problem = LinearVariationalProblem(stokes(phi, INMOUTH1), L, U2, bcs=bcs2)
    solver_stokes2 = LinearVariationalSolver(problem, solver_parameters=stokes_parameters)
    solver_stokes2.solve()
    print("Iterations Stokes 2 # {}".format(solver_stokes2.snes.ksp.getIterationNumber()))
    u2, _ = U2.split()
    File("u2.pvd").write(u2)

    # Convection difussion equation
    ks = Constant(1e0)
    cp_value = 5.0e3
    cp = Constant(cp_value)
    T = FunctionSpace(mesh, 'DG', 1)
    t = Function(T, name="Temperature")
    w = TestFunction(T)

    # Mesh-related functions
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    u1, p1 = split(U1)
    u2, p2 = split(U2)

    def upwind(u):
        return (dot(u, n) + abs(dot(u, n)))/2.0
    u1n = upwind(u1)
    u2n = upwind(u2)

    # Penalty term
    alpha = Constant(50000.0) # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
    alpha = Constant(500.0) # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
    # Bilinear form
    a_int = dot(grad(w), ks*grad(t) - cp*(u1 + u2)*t)*dx

    a_fac = Constant(-1.0)*ks*dot(avg(grad(w)), jump(t, n))*dS \
        + Constant(-1.0)*ks*dot(jump(w, n), avg(grad(t)))*dS \
          + ks('+')*(alpha('+')/avg(h))*dot(jump(w, n), jump(t, n))*dS

    a_vel = dot(jump(w), cp*(u1n('+') + u2n('+'))*t('+') - \
            cp*(u1n('-') + u2n('-'))*t('-'))*dS + \
            dot(w, cp*(u1n + u2n)*t)*ds

    a_bnd = dot(w, cp*dot(u1 + u2, n)*t)*(ds(INLET1) + ds(INLET2)) \
            + w*t*(ds(INLET1) + ds(INLET2)) \
            - w*tin1*ds(INLET1) \
            - w*tin2*ds(INLET2) \
            + alpha/h * ks *w*t*(ds(INLET1) + ds(INLET2)) \
            - ks * dot(grad(w), t*n)*(ds(INLET1) + ds(INLET2)) \
            - ks * dot(grad(t), w*n)*(ds(INLET1) + ds(INLET2))

    aT = a_int + a_fac + a_vel + a_bnd

    LT_bnd = alpha/h * ks * tin1 * w * ds(INLET1)  \
            + alpha/h * ks * tin2 * w * ds(INLET2) \
            - tin1 * ks * dot(grad(w), n) * ds(INLET1) \
            - tin2 * ks * dot(grad(w), n) * ds(INLET2)
    eT = aT - LT_bnd

    problem = NonlinearVariationalProblem(eT, t)
    solver_temp = NonlinearVariationalSolver(problem, solver_parameters=temperature_parameters)
    solver_temp.solve()
    print("Iterations Temperature # {}".format(solver_temp.snes.ksp.getIterationNumber()))
    File("t.pvd").write(t)

if __name__ == '__main__':

    import os
    import sys
    if len(sys.argv) > 1:
        files = [sys.argv[1]]
    else:
        files = os.listdir("./preconditioners/")

    for file in files:
        if file[-3:] != '.py':
            continue

        import importlib
        try:
            mesh = getattr(importlib.import_module("preconditioners." + file[:-3]), "mesh")
            stokes_parameters = getattr(importlib.import_module("preconditioners." + file[:-3]), "stokes_parameters")
            temperature_parameters = getattr(importlib.import_module("preconditioners." + file[:-3]), "temperature_parameters")
        except:
            print('Invalid python file')
        print("Solver {}".format(file[:-3]))
        main(mesh, stokes_parameters, temperature_parameters)
