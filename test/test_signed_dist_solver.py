import sys

sys.path.append("../")
from lestofire.optimization import ReinitSolverDG
from firedrake import (
    UnitSquareMesh,
    FunctionSpace,
    Expression,
    interpolate,
    File,
    errornorm,
    Function,
    DirichletBC,
    Constant,
    SpatialCoordinate,
    H1,
    sqrt,
)
from ufl import sin
import pytest

N = 50


@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(N, N)


@pytest.fixture(scope="module")
def DG0(mesh):
    return FunctionSpace(mesh, "DG", 0)


@pytest.mark.parametrize(("iterative"), [False])
def test_cone(mesh, DG0, iterative):

    solver = ReinitSolverDG(mesh, n_steps=100, dt=5e-3)

    radius = 0.2
    x, y = SpatialCoordinate(mesh)
    phi_init = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) - radius * radius
    phi0 = Function(DG0).interpolate(phi_init)

    phi_solution = interpolate(
        sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)) - radius, DG0
    )
    phin = solver.solve(phi0)
    error = errornorm(phin, phi_solution)
    print(f"Error: {error}")

    error_dg_after_50 = 0.010352889031168062
    assert pytest.approx(error, 1e-8) == error_dg_after_50


@pytest.mark.parametrize(("iterative"), [False])
def test_parabole(mesh, DG0, iterative):

    solver = ReinitSolverDG(mesh, n_steps=100, dt=5e-3)

    radius = 0.2
    x, y = SpatialCoordinate(mesh)
    phi_init = (x - 0.5) * (x - 0.5) - radius * radius
    phi0 = Function(DG0).interpolate(phi_init)

    phi_solution = interpolate(sqrt((x - 0.5) * (x - 0.5)) - radius, DG0)
    phin = solver.solve(phi0)
    error = errornorm(phin, phi_solution)
    print(f"Error: {error}")

    error_dg_after_50 = 0.006539568004459555
    assert pytest.approx(error, 1e-8) == error_dg_after_50