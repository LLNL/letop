import sys

sys.path.append("../")
import pytest
from firedrake import (
    H1,
    File,
    Function,
    FunctionSpace,
    Mesh,
    SpatialCoordinate,
    UnitSquareMesh,
    errornorm,
    interpolate,
    sqrt,
)
from lestofire.optimization import ReinitSolverDG
from ufl import sin

N = 50


@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(N, N)


@pytest.mark.parametrize(
    "test_mesh,x_shift,error, p",
    [
        (UnitSquareMesh(N, N, diagonal="right"), 0.0, 0.05192113964921833, 0),
        (Mesh("./unstructured_rectangle.msh"), 0.0, 0.05494971697014042, 0),
        (UnitSquareMesh(N, N, diagonal="right"), 0.5, 0.02200988416607167, 0),
        (Mesh("./unstructured_rectangle.msh"), 0.5, 0.019432037920418723, 0),
        (UnitSquareMesh(N, N, diagonal="right"), 0.0, 0.05672086032767158, 1),
        (Mesh("./unstructured_rectangle.msh"), 0.0, 0.05464336575646338, 1),
        (UnitSquareMesh(N, N, diagonal="right"), 0.5, 0.023615517657939632, 1),
        (Mesh("./unstructured_rectangle.msh"), 0.5, 0.018194548256388755, 1),
    ],
)
def test_cone(test_mesh, x_shift, error, p):

    DGp = FunctionSpace(test_mesh, "DG", p)

    if p > 0:
        target_continuity = H1
    else:
        target_continuity = None
    phi_pvd = File("reinit.pvd", target_continuity=target_continuity)

    solver = ReinitSolverDG(n_steps=200, dt=2e-3, h_factor=5.0, phi_pvd=phi_pvd)

    radius = 0.2
    x, y = SpatialCoordinate(test_mesh)
    phi_init = (x - x_shift) * (x - x_shift) + (y - 0.5) * (y - 0.5) - radius * radius
    phi0 = Function(DGp).interpolate(phi_init)

    phi_solution = interpolate(
        sqrt((x - x_shift) * (x - x_shift) + (y - 0.5) * (y - 0.5)) - radius, DGp
    )
    phin = solver.solve(phi0)
    error_numeri = errornorm(phin, phi_solution)
    assert pytest.approx(error, 1e-4) == error_numeri
