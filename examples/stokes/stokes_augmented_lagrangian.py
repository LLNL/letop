from firedrake import *
from firedrake_adjoint import *
import numpy as np
from ufl import replace, conditional
from ufl import min_value, max_value

from mesh_stokes_flow import INMOUTH1, INMOUTH2, OUTMOUTH1, OUTMOUTH2, WALLS

from lestofire import LevelSetLagrangian, AugmentedLagrangianOptimization, RegularizationSolver

def main():
    output_dir = "stokes_levelset_darcy/"

    mesh = Mesh("./mesh_stokes.msh")

    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    x, y = SpatialCoordinate(mesh)
    PHI = FunctionSpace(mesh, 'CG', 1)
    phi_expr = sin(y*pi/0.2)*cos(x*pi/0.2) - Constant(0.8)

    block_inlet = True
    if block_inlet:
        scaling = Constant(-1.0)
    else:
        scaling = Constant(1.0)

    phi_expr *= scaling


    phi = interpolate(phi_expr , PHI)
    phi.rename("LevelSet")
    File(output_dir + "phi_initial.pvd").write(phi)

    mu = Constant(1e2)
    alphamax = 1e4
    alphamin = 1e-12
    epsilon = Constant(100000.0)
    def hs(phi, epsilon):
        return Constant(1.0) / ( Constant(1.0) + exp(-epsilon*phi)) + Constant(alphamin)

    P2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2*P1
    W = FunctionSpace(mesh, TH)

    U = Function(W, name='Solution')
    u, p = split(U)
    V = TestFunction(W)
    v, q = split(V)

    f = Constant((0.0, 0.0))
    a = mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u)
    darcy_term = Constant(alphamax)*hs(phi, epsilon)*inner(u, v)
    L = inner(f, v)*dx

    e1f = a*dx + darcy_term*dx

    inflow = Constant((5.0, 0.0))
    noslip = Constant((0.0, 0.0))
    bc1 = DirichletBC(W.sub(0), noslip, WALLS)
    bc2 = DirichletBC(W.sub(0), inflow, 1)
    bc3 = DirichletBC(W.sub(0), inflow, 2)
    bc4 = DirichletBC(W.sub(1), Constant(0.0), 3)
    bc5 = DirichletBC(W.sub(1), Constant(0.0), 4)
    bcs=[bc1, bc2, bc3, bc4, bc5]

    parameters = {
            'ksp_type':'preonly', 'pc_type':'lu',
            "mat_type": "aij",
            'ksp_converged_reason' : None,
            "pc_factor_mat_solver_type": "mumps"
            }
    a = derivative(e1f, U)
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

    solve(a==L, U, bcs, solver_parameters=parameters)#, nullspace=nullspace)
    u, p = split(U)

    penalty = 8e6
    with stop_annotating():
        total_area = assemble(Constant(1.0)*dx(domain=mesh))

    VolPen = assemble((hs(-phi, epsilon)*Constant(1.0) - Constant(0.3*total_area))*dx)

    Jform = assemble(Constant(alphamax)*hs(phi, epsilon)*inner(mu*u, u)*dx \
                + mu/Constant(2.0)*inner(grad(u), grad(u))*dx)

    print("Initial cost function value {}".format(Jform))
    phi_pvd = File(output_dir + "phi_evolution.pvd")
    def deriv_cb(phi):
        phi_pvd.write(phi[0])

    c = Control(s)
    Jhat = LevelSetLagrangian(Jform, c, phi, derivative_cb_pre=deriv_cb, constraint=VolPen, method='AL')
    Jhat_v = Jhat(phi)
    dJ = Jhat.derivative()

    velocity = Function(S)
    bcs_vel_1 = DirichletBC(S, noslip, 1)
    bcs_vel_2 = DirichletBC(S, noslip, 2)
    bcs_vel_3 = DirichletBC(S, noslip, 3)
    bcs_vel_4 = DirichletBC(S, noslip, 4)
    bcs_vel_5 = DirichletBC(S, noslip, 5)
    bcs_vel = [bcs_vel_1, bcs_vel_2, bcs_vel_3, bcs_vel_4, bcs_vel_5]
    reg_solver = RegularizationSolver(S, mesh, beta=1e4, gamma=0.0, dx=dx, bcs=bcs_vel)

    #Jhat.optimize_tape()
    tape = get_working_tape()
    tape.visualise()

    options = {
             'hmin' : 0.01414,
             'hj_stab': 1.5,
             'dt_scale' : 0.1,
             'n_hj_steps' : 3,
             'max_iter' : 30
             }


    opti_solver = AugmentedLagrangianOptimization(Jhat, reg_solver, options=options, pvd_output=phi_pvd)
    Jarr = opti_solver.solve(phi, velocity, solver_parameters=parameters)

    from numpy.testing import assert_allclose
    assert_allclose(Jarr[14], 1834899.78206, rtol=1e-3, atol=1e-6, err_msg='Optimization broken')

    # It 14 is 1835676.2021


if __name__ == '__main__':
    main()
