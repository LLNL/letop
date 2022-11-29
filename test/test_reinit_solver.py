import firedrake as fd
from firedrake import sqrt
from letop.optimization import ReinitSolverCG
from letop.physics import max_mesh_dimension
import pytest


def test_max_dim():
    mesh = fd.UnitSquareMesh(100, 100)

    assert pytest.approx(max_mesh_dimension(mesh), 1.0)

    mesh = fd.RectangleMesh(10, 20, 5.0, 3.0)

    assert pytest.approx(max_mesh_dimension(mesh), 5.0)

    mesh = fd.CubeMesh(10, 10, 10, 8.0)

    assert pytest.approx(max_mesh_dimension(mesh), 8.0)


def test_cone_cg_2D():
    mesh = fd.UnitSquareMesh(100, 100)
    V = fd.FunctionSpace(mesh, "CG", 1)

    radius = 0.2
    x, y = fd.SpatialCoordinate(mesh)
    x_shift = 0.5
    phi_init = (
        (x - x_shift) * (x - x_shift) + (y - 0.5) * (y - 0.5) - radius ** 2
    )
    phi0 = fd.Function(V).interpolate(phi_init)

    phi_pvd = fd.File("phi_reinit.pvd")

    phi = fd.Function(V)
    reinit_solver = ReinitSolverCG(phi)
    phin = reinit_solver.solve(phi0, iters=10)
    phi_pvd.write(phin)

    phi_solution = fd.interpolate(
        sqrt((x - x_shift) * (x - x_shift) + (y - 0.5) * (y - 0.5)) - radius,
        V,
    )
    error_numeri = fd.errornorm(phin, phi_solution)
    print(f"error: {error_numeri}")
    assert error_numeri < 1e-4
