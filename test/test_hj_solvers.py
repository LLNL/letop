from firedrake import UnitSquareMesh, FunctionSpace, Expression, \
                    interpolate, File, errornorm, Function,  \
                    DirichletBC, Constant, SpatialCoordinate, \
                    FiniteElement, VectorFunctionSpace
from ufl import as_vector
import pytest
import sys
sys.path.append("../")
from lestofire.optimization import HJStabSolver, HJSUPG, HJDG

N = 100
@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(N, N)

@pytest.fixture(scope='module')
def phicg1_elem(mesh):
    return FiniteElement('CG', mesh.ufl_cell(), 1)

@pytest.fixture(scope='module')
def phicg1(mesh, phicg1_elem):
    return FunctionSpace(mesh, phicg1_elem)

@pytest.fixture(scope='module')
def phicg1_vector(mesh, phicg1_elem):
    return VectorFunctionSpace(mesh, phicg1_elem)

@pytest.fixture(scope='module')
def X(mesh):
    return SpatialCoordinate(mesh)

@pytest.fixture(scope='function')
def phi_x0(X):
    phi_x0 = (X[0] < 0.2)*(X[1] > 0.5)
    return phi_x0



def time_loop(mesh, X, hj_solver, phicg1, phicg1_vector):
    t = 0.0
    beta = interpolate(as_vector([1.0, 0.0]), phicg1_vector)
    phi_expr = (X[0] - t < 0.2)*(X[1] > 0.5)
    phi_n = interpolate(phi_expr, phicg1)
    import numpy as np
    hmin = np.sqrt(2.0 / (N*N)) # TODO, hardcoded for UnitSquareMesh
    for i in range(50):
        # CFL step
        phi_expr = (X[0] - t < 0.2)*(X[1] > 0.5)
        maxv = np.max(phi_n.vector()[:])
        dt = hmin / maxv

        # Solve
        phi_next = hj_solver.solve(beta, phi_n, steps=1, t=t, dt=dt)

        phi_n.assign(phi_next)
        t += dt

        phi_exact = interpolate(phi_expr, phicg1)
        error_phi = errornorm(phi_exact, phi_n)
        print("Error: {0:.12f}".format(error_phi))
    return error_phi

@pytest.mark.parametrize(('iterative'), [False, True])
def test_stab_solver(mesh, phicg1, phi_x0, X, phicg1_vector, iterative):
    bc = DirichletBC(phicg1, phi_x0, 1)
    hj_solver = HJStabSolver(mesh, phicg1, c2_param=0.2, bc=bc, iterative=iterative)
    error_phi = time_loop(mesh, X, hj_solver,phicg1, phicg1_vector)

    error_stab_after_50 = 0.08070355337702109
    assert pytest.approx(error_phi, 1e-5) == error_stab_after_50

def test_supg_solver(mesh, phicg1, phi_x0, X, phicg1_vector):
    bc = DirichletBC(phicg1, phi_x0, 1)
    hj_solver = HJSUPG(mesh, phicg1, bc=bc)
    error_phi = time_loop(mesh, X, hj_solver, phicg1, phicg1_vector)

    error_supg_after_50 = 0.04416640448327516
    assert pytest.approx(error_phi, 1e-8) == error_supg_after_50
