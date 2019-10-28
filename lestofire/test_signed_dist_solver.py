from optimization import SignedDistanceSolver
from firedrake import UnitSquareMesh, FunctionSpace, Expression, \
                    interpolate, File, errornorm, Function,  \
                    DirichletBC, Constant

import pytest

@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(100,100)


@pytest.fixture(scope='module')
def phicg1(mesh):
    return FunctionSpace(mesh, 'CG', 1)

def test_reinit(mesh, phicg1):

    solver = SignedDistanceSolver(mesh, phicg1, dt=1e-6, n_steps=100)

    Dx = mesh.hmin()
    phi0 = interpolate(Expression("sin(x[1]/0.1)*sin(x[0]/0.1) - 0.5", element=phicg1.ufl_element()), phicg1)
    phi1 = solver.solve(phi0, Dx)

    phi0 = interpolate(Expression("sin(x[1]/0.1)*sin(x[0]/0.1) - 0.5", element=phicg1.ufl_element()), phicg1)
    error = errornorm(phi0, phi1)

    assert pytest.approx(error, 1e-8) == 0.05106448677447494
