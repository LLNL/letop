from firedrake import *
from firedrake_adjoint import *

from lestofire import LevelSetLagrangian, AugmentedLagrangianOptimization, RegularizationSolver, SteepestDescent
import pytest

def test_augmented_lagrangian():

    mesh = UnitSquareMesh(100, 100)

    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    x, y = SpatialCoordinate(mesh)
    PHI = FunctionSpace(mesh, 'CG', 1)
    phi_expr = sin(y*pi/0.2)*cos(x*pi/0.2) - Constant(0.8)

    phi = interpolate(phi_expr , PHI)
    phi.rename("LevelSet")

    alphamin = 1e-12
    epsilon = Constant(100000.0)
    def hs(phi, epsilon):
        return Constant(1.0) / ( Constant(1.0) + exp(-epsilon*phi)) + Constant(alphamin)

    scaling = -100.0
    Jform = assemble(Constant(scaling)*hs(phi, epsilon)*x*dx)
    print("Initial cost function value {}".format(Jform))
    VolPen = assemble(hs(phi, epsilon)*dx(domain=mesh) - Constant(0.5)*dx(domain=mesh))

    phi_pvd = File("phi_evolution.pvd")
    def deriv_cb(phi):
        phi_pvd.write(phi[0])

    c = Control(s)
    Jhat = LevelSetLagrangian(Jform, c, phi, lagrange_multiplier=50.0, penalty_value=10.0, penalty_update=2.0, derivative_cb_pre=deriv_cb, constraint=VolPen, method='AL')
    reg_solver = RegularizationSolver(S, mesh, beta=1e-1, gamma=1.0, dx=dx)

    options = {
             'hmin' : 0.01414,
             'hj_stab': 5.0,
             'dt_scale' : 1e-2,
             'n_hj_steps' : 3,
             'max_iter' : 30,
             'n_reinit' : 5,
             'stopping_criteria' : 1e-2
             }

    parameters = {
            'ksp_type':'preonly', 'pc_type':'lu',
            "mat_type": "aij",
            'ksp_converged_reason' : None,
            "pc_factor_mat_solver_type": "mumps"
            }


    velocity = Function(S)
    opti_solver = AugmentedLagrangianOptimization(Jhat, reg_solver, options=options, pvd_output=phi_pvd)
    Jarr = opti_solver.solve(phi, velocity, iterative=False)

    from numpy.testing import assert_allclose
    analytical_solution = 1.0/2.0 - 1.0/8.0

    assert_allclose(Jarr[Jarr != 0][-1], scaling*analytical_solution, rtol=1e-2, err_msg='Optimization broken')
