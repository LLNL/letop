import sys
from firedrake.mg.mesh import MeshHierarchy

from firedrake.utility_meshes import RectangleMesh

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
    Mesh,
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


@pytest.mark.parametrize(
    ("mesh_type"),
    [UnitSquareMesh(N, N, diagonal="right")],
)
def test_shifted_cone(mesh_type):

    DG0 = FunctionSpace(mesh_type, "DG", 0)

    solver = ReinitSolverDG(mesh_type, n_steps=100, dt=5e-3)

    radius = 0.2
    x, y = SpatialCoordinate(mesh_type)
    phi_init = (x) * (x) + (y - 0.5) * (y - 0.5) - radius * radius
    phi0 = Function(DG0).interpolate(phi_init)

    phi_solution = interpolate(sqrt((x) * (x) + (y - 0.5) * (y - 0.5)) - radius, DG0)
    phin = solver.solve(phi0)
    error = errornorm(phin, phi_solution)
    print(f"Error: {error}")

    error_dg_after_50 = 0.010352889031168062
    assert pytest.approx(error, 1e-8) == error_dg_after_50


def test_cone(mesh_type):

    DG0 = FunctionSpace(mesh_type, "DG", 0)

    solver = ReinitSolverDG(mesh_type, n_steps=100, dt=5e-3)

    radius = 0.2
    x, y = SpatialCoordinate(mesh_type)
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


@pytest.mark.parametrize(("iterative"), [False])
def test_compliance_initial_level_set(iterative):
    from ufl import cos, max_value, pi

    lx, ly = 2.0, 1.0
    Nx, Ny = 202, 101
    # mesh = Mesh("./mesh_cantilever.msh")
    # mesh = RectangleMesh(Nx, Ny, lx, ly, quadrilateral=True)
    mesh = RectangleMesh(Nx, Ny, lx, ly, diagonal="crossed")
    DG0 = FunctionSpace(mesh, "DG", 0)
    x, y = SpatialCoordinate(mesh)
    phi_init = (
        -cos(8.0 / lx * pi * x) * cos(4.0 * pi * y)
        - 0.4
        # + max_value(50.0 * (0.01 - x ** 2 - (y - ly / 2) ** 2), 0.0)
        + max_value(100.0 * (x + y - lx - ly + 0.1), 0.0)
        + max_value(100.0 * (x - y - lx + 0.1), 0.0)
    )

    dt = 0.2 * lx / Nx
    solver = ReinitSolverDG(mesh, n_steps=2000, dt=dt)

    x, y = SpatialCoordinate(mesh)
    phi0 = Function(DG0).interpolate(phi_init)

    phin = solver.solve(phi0)