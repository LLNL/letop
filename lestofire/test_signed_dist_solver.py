from .optimization import SignedDistanceSolver
from firedrake import UnitSquareMesh, FunctionSpace, Expression, \
                    interpolate, File, errornorm, Function,  \
                    DirichletBC, Constant, SpatialCoordinate
from ufl import sin
import pytest

N = 100
@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(N, N)

@pytest.fixture(scope='module')
def phicg1(mesh):
    return FunctionSpace(mesh, 'CG', 1)

def test_reinit(mesh, phicg1):

    solver = SignedDistanceSolver(mesh, phicg1, dt=1e-6, n_steps=100)
    X = SpatialCoordinate(mesh)

    import numpy as np
    Dx = np.sqrt(2.0 / (N*N)) # TODO, hardcoded for UnitSquareMesh
    phi0expr = sin(X[1]/0.1)*sin(X[0]/0.1) - 0.5
    phi0 = interpolate(phi0expr, phicg1)
    phi1 = solver.solve(phi0, Dx)

    phi0expr = sin(X[1]/0.1)*sin(X[0]/0.1) - 0.5
    phi0 = interpolate(phi0expr, phicg1)
    error = errornorm(phi0, phi1)

    assert pytest.approx(error, 1e-8) == 0.05106474186151434
