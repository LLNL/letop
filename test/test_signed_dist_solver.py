import sys
from firedrake.mg.mesh import MeshHierarchy


sys.path.append("../")
from lestofire.optimization import ReinitSolverDG
from firedrake import (
    UnitSquareMesh,
    FunctionSpace,
    interpolate,
    errornorm,
    Function,
    Mesh,
    SpatialCoordinate,
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
    "test_mesh,x_shift,error",
    [
        (UnitSquareMesh(N, N, diagonal="right"), 0.0, 0.05192113964921833),
        (Mesh("./unstructured_rectangle.msh"), 0.0, 0.05494971697014042),
        (UnitSquareMesh(N, N, diagonal="right"), 0.5, 0.02200988416607167),
        (Mesh("./unstructured_rectangle.msh"), 0.5, 0.019432037920418723),
    ],
)
def test_cone(test_mesh, x_shift, error):

    DG0 = FunctionSpace(test_mesh, "DG", 0)

    solver = ReinitSolverDG(test_mesh, n_steps=200, dt=2e-3)

    radius = 0.2
    x, y = SpatialCoordinate(test_mesh)
    phi_init = (x - x_shift) * (x - x_shift) + (y - 0.5) * (y - 0.5) - radius * radius
    phi0 = Function(DG0).interpolate(phi_init)

    phi_solution = interpolate(
        sqrt((x - x_shift) * (x - x_shift) + (y - 0.5) * (y - 0.5)) - radius, DG0
    )
    phin = solver.solve(phi0)
    error_numeri = errornorm(phin, phi_solution)
    assert pytest.approx(error, 1e-8) == error_numeri


@pytest.mark.parametrize(("iterative"), [False])
def test_compliance_initial_level_set(iterative):
    from ufl import cos, max_value, pi

    lx, ly = 2.0, 1.0
    Nx, Ny = 202, 101
    m = Mesh("./mesh_cantilever.msh")
    mesh = MeshHierarchy(m, 1)[-1]
    # mesh = RectangleMesh(Nx, Ny, lx, ly, quadrilateral=True)
    # mesh = RectangleMesh(Nx, Ny, lx, ly, diagonal="crossed")
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