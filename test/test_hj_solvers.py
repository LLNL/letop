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
from firedrake.bcs import DirichletBC
from firedrake.functionspace import VectorFunctionSpace
import pytest
import sys
import numpy as np
from ufl import sin, pi

sys.path.append("../")
from lestofire.optimization import HJLocalDG

N = 100


def time_loop(hj_solver, phi_expr, phi0):
    t = 0.0
    V = phi0.function_space()
    mesh = V.ufl_domain()
    beta = interpolate(Constant((1.0, 0.0)), VectorFunctionSpace(mesh, "DG", 0))

    phi_n = interpolate(phi_expr(0.0), V)
    phi_exact = interpolate(phi_expr(0.0), V)

    for i in range(50):

        # Solve
        phi0.interpolate(phi_expr(t))
        phi_next = hj_solver.solve(beta, phi_n, steps=1, scaling=0.1)

        phi_n.assign(phi_next)
        t += hj_solver.dt

        phi_exact.interpolate(phi_expr(t))
        error_phi = errornorm(phi_exact, phi_n)
    return error_phi


def test_dg():

    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "DG", 0)
    hmin = np.sqrt(2.0 / (N * N))

    X = SpatialCoordinate(mesh)

    def phi_expr(t):
        return sin((X[0] - t) * pi / 0.5)

    phi0 = interpolate(phi_expr(0.0), V)
    bcs = DirichletBC(V, phi0, (1,), method="geometric")
    hj_solver = HJLocalDG(mesh, V, bcs=bcs, hmin=hmin)
    error_phi = time_loop(hj_solver, phi_expr, phi0)

    error_dg_after_50 = 0.006192889800985826
    assert pytest.approx(error_phi, 1e-8) == error_dg_after_50
