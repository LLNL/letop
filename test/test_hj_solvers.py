from firedrake import (
    UnitSquareMesh,
    FunctionSpace,
    interpolate,
    File,
    errornorm,
    Constant,
    SpatialCoordinate,
    FiniteElement,
    H1,
    ds,
)
from firedrake.functionspace import VectorFunctionSpace
from ufl import as_vector
import pytest
import sys
import numpy as np
from ufl import sin, pi

sys.path.append("../")
from lestofire.optimization import HJDG

N = 100


@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(N, N)


@pytest.fixture(scope="module")
def phidg0_elem(mesh):
    return FiniteElement("DG", mesh.ufl_cell(), 0)


@pytest.fixture(scope="module")
def phidg0(mesh, phicg1_elem):
    return FunctionSpace(mesh, phidg0_elem)


@pytest.fixture(scope="module")
def X(mesh):
    return SpatialCoordinate(mesh)


def time_loop(mesh, X, hj_solver, phidg0, phi_expr):
    t = 0.0
    beta = interpolate(Constant((1.0, 0.0)), VectorFunctionSpace(mesh, "DG", 0))

    phi_n = interpolate(phi_expr(0.0), phidg0)
    phi_pvd = File("test_hjdg.pvd", target_continuity=H1)
    phi_exact_pvd = File("test_exact_hjdg.pvd", target_continuity=H1)
    phi_exact = interpolate(phi_expr(0.0), phidg0)

    for i in range(50):

        # Solve
        phi_next = hj_solver.solve(beta, phi_n, steps=1, t=t, scaling=0.2)

        phi_n.assign(phi_next)
        phi_pvd.write(phi_next)
        t += hj_solver.dt

        phi_exact.interpolate(phi_expr(t))
        phi_exact_pvd.write(phi_exact)
        error_phi = errornorm(phi_exact, phi_n)
        print("Error: {0:.12f}".format(error_phi))
    return error_phi


def test_dg(mesh, X):
    from dolfin_dg import DGDirichletBC

    V = FunctionSpace(mesh, "DG", 0)
    hmin = np.sqrt(2.0 / (N * N))  # TODO, hardcoded for UnitSquareMesh

    def phi_expr(t):
        return X[0] * sin((X[0] - t) * pi / 0.5)

    phi0 = phi_expr(0.0)
    phi_x0 = interpolate(phi0, V)
    bcs = DGDirichletBC(ds(1), phi0)
    hj_solver = HJDG(mesh, V, phi_x0, bcs=bcs, hmin=hmin)
    error_phi = time_loop(mesh, X, hj_solver, V, phi_expr)

    error_dg_after_50 = 0.010686410367
    assert pytest.approx(error_phi, 1e-8) == error_dg_after_50
