import sys

import numpy as np
import pytest
from firedrake import (
    H1,
    Constant,
    File,
    FiniteElement,
    FunctionSpace,
    SpatialCoordinate,
    UnitSquareMesh,
    ds,
    errornorm,
    interpolate,
)
from firedrake.bcs import DirichletBC
from firedrake.functionspace import VectorFunctionSpace
from ufl import pi, sin

sys.path.append("../")
from lestofire.optimization import HJLocalDG

N = 100


def time_loop(hj_solver, phi_expr, phi0):
    t = 0.0
    V = phi0.function_space()
    mesh = V.ufl_domain()
    beta = interpolate(
        Constant((1.0, 0.0)), VectorFunctionSpace(mesh, "DG", V.ufl_element().degree())
    )

    phi_n = interpolate(phi_expr(0.0), V)
    phi_exact = interpolate(phi_expr(0.0), V)

    for i in range(50):

        # Solve
        phi0.interpolate(phi_expr(t))
        phi_next = hj_solver.solve(beta, phi_n, scaling=0.1)

        phi_n.assign(phi_next)
        t += hj_solver.dt

        phi_exact.interpolate(phi_expr(t))
        error_phi = errornorm(phi_exact, phi_n)
    return error_phi


@pytest.mark.parametrize(
    "error, p",
    [
        (0.006192889800985826, 0),
        (0.0032936167264323423, 1),
    ],
)
def test_dg(error, p):

    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, "DG", p)
    hmin = np.sqrt(2.0 / (N * N))

    X = SpatialCoordinate(mesh)

    def phi_expr(t):
        return sin((X[0] - t) * pi / 0.5)

    phi0 = interpolate(phi_expr(0.0), V)
    bcs = DirichletBC(V, phi0, (1,), method="geometric")
    hj_solver = HJLocalDG(bcs=bcs, hmin=hmin, n_steps=1)
    error_phi = time_loop(hj_solver, phi_expr, phi0)

    assert pytest.approx(error_phi, 1e-8) == error
