from firedrake import *
from firedrake_adjoint import *

from lestofire import LevelSetLagrangian, AugmentedLagrangianOptimization, RegularizationSolver, SteepestDescent
import pytest
from numpy.testing import assert_allclose

def test_augmented_lagrangian():

    mesh = UnitSquareMesh(100, 100)

    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    x, y = SpatialCoordinate(mesh)
    PHI = FunctionSpace(mesh, 'CG', 1)
    phi_expr = -x + 0.5

    phi_pvd = File("phi_evolution.pvd")
    phi = interpolate(phi_expr , PHI)
    phi.rename("LevelSet")
    phi_pvd.write(phi)

    alphamin = 1e-12
    epsilon = Constant(100000.0)
    def hs(phi, epsilon):
        return Constant(1.0) / ( Constant(1.0) + exp(-epsilon*phi)) + Constant(alphamin)

    Jform = assemble(hs(phi, epsilon)*dx)
    # Value for Jform should be 0.5
    assert_allclose(Jform, 0.5, rtol=1e-4, err_msg='Optimization broken')

    print("Initial cost function value {}".format(Jform))
    VolPen = assemble(hs(phi, epsilon)*dx(domain=mesh) - Constant(0.2)*dx(domain=mesh))
    # Value for Jform should be 0.3
    assert_allclose(VolPen, 0.3, rtol=1e-4, err_msg='Optimization broken')

    def deriv_cb(phi):
        phi_pvd.write(phi[0])

    c = Control(s)
    Jhat = LevelSetLagrangian(Jform, c, phi, lagrange_multiplier=50.0, penalty_value=10.0, penalty_update=2.0, derivative_cb_pre=deriv_cb, constraint=VolPen, method='AL')
    # Value for Jhat is Jform + lagrange_multiplier*VolPen +
    # penalty_value*VolPen*VolPen * 1.0/2.0 = 0.5 + 50.0*0.3 +
    # 1.0/2.0*10.0*0.3*0.3
    assert_allclose(Jhat(phi), 15.95, rtol=1e-4, err_msg='Optimization broken')

    # Update penalty: 10*2.0, therefore Jhat = 0.5 + 50.0*0.3 +
    # 1.0/2.0*20.0*0.3*0.3 = 16.4
    Jhat.update_penalty()
    assert_allclose(Jhat(phi), 16.4, rtol=1e-4, err_msg='Optimization broken')

    # With the new penalty, the lagrangian should be: 50.0 + 20.0 * 0.3 = 56.0
    # Therefore Jhat = 0.5 + 56.0*0.3 + 1.0/2.0*20.0*0.3*0.3 =
    Jhat.update_lagrangian()
    assert_allclose(Jhat(phi), 18.2, rtol=1e-4, err_msg='Optimization broken')

