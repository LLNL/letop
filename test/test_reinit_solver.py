import firedrake as fd
from firedrake import sqrt
from lestofire.optimization import ReinitializationSolver
from lestofire.physics import max_mesh_dimension
import pytest


N = 50


def test_cone_3D():
    mesh = fd.UnitCubeMesh(20, 20, 20)
    DGp = fd.FunctionSpace(mesh, "DG", 1)
    phi_pvd = fd.File("reinit.pvd", target_continuity=fd.H1)
    phi_sol = fd.Function(DGp, name="solution")

    from itertools import count

    global_counter = count()

    def monitor(ts, i, t, x):
        iter = next(global_counter)
        if iter % 5 == 0:
            with phi_sol.dat.vec as v:
                x.copy(v)
            phi_pvd.write(phi_sol)

    solver_parameters = {
        "ts_type": "rk",
        "ts_rk_type": "5dp",
        "ts_atol": 1e-5,
        "ts_rtol": 1e-5,
        "ts_dt": 1e-3,
        "ts_converged_reason": None,
        "ts_monitor": None,
        "ts_adapt_type": "dsp",
        "ts_exact_final_time": "matchstep",
        "h_factor": 5,
        "stopping_criteria": 1.0,
    }
    solver = ReinitializationSolver(
        DGp,
        monitor_callback=monitor,
        poststep=False,
        solver_parameters=solver_parameters,
    )

    radius = 0.2
    x, y, z = fd.SpatialCoordinate(mesh)
    x_shift = 0.5
    phi_init = (
        (x - x_shift) * (x - x_shift)
        + (y - 0.5) * (y - 0.5)
        + (z - 0.5) * (z - 0.5)
        - radius * radius
    )
    phi0 = fd.Function(DGp).interpolate(phi_init)

    phin = solver.solve(phi0, total_t=0.4)
    phi_solution = fd.interpolate(
        sqrt(
            (x - x_shift) * (x - x_shift)
            + (y - 0.5) * (y - 0.5)
            + (z - 0.5) * (z - 0.5)
        )
        - radius,
        DGp,
    )
    error_numeri = fd.errornorm(phin, phi_solution)
    print(f"error: {error_numeri}")
    assert pytest.approx(0.06605266582620326, 1e-4) == error_numeri


@pytest.mark.parametrize(
    "test_mesh,x_shift,error, p",
    [
        (fd.UnitSquareMesh(N, N, diagonal="right"), 0.0, 0.051679991911252984, 0),
        (fd.Mesh("./unstructured_rectangle.msh"), 0.0, 0.05467660491552649, 0),
        (fd.UnitSquareMesh(N, N, diagonal="right"), 0.5, 0.022107216282791796, 0),
        (fd.Mesh("./unstructured_rectangle.msh"), 0.5, 0.019569327325334684, 0),
        (fd.UnitSquareMesh(N, N, diagonal="right"), 0.0, 0.056456035431546446, 1),
        (fd.Mesh("./unstructured_rectangle.msh"), 0.0, 0.05434256956234183, 1),
        (fd.UnitSquareMesh(N, N, diagonal="right"), 0.5, 0.02372197275119883, 1),
        (fd.Mesh("./unstructured_rectangle.msh"), 0.5, 0.01832726713117936, 1),
        (fd.UnitSquareMesh(12, 12, quadrilateral=True), 0.5, 0.05845317085436576, 2),
    ],
)
def test_cone(test_mesh, x_shift, error, p):

    DGp = fd.FunctionSpace(test_mesh, "DG", p)

    phi_pvd = fd.File("reinit.pvd", target_continuity=fd.H1)
    phi_sol = fd.Function(DGp, name="solution")

    def monitor(ts, i, t, x):
        with phi_sol.dat.vec as v:
            x.copy(v)
        phi_pvd.write(phi_sol)

    solver_parameters = {
        "ts_type": "rk",
        "ts_rk_type": "5dp",
        "ts_atol": 1e-5,
        "ts_rtol": 1e-5,
        "ts_dt": 1e-3,
        "ts_converged_reason": None,
        "ts_monitor": None,
        "ts_adapt_type": "dsp",
        "ts_exact_final_time": "matchstep",
        "h_factor": 5,
        "stopping_criteria": 5.0,
    }
    solver = ReinitializationSolver(
        DGp,
        monitor_callback=monitor,
        poststep=False,
        solver_parameters=solver_parameters,
    )

    radius = 0.2
    x, y = fd.SpatialCoordinate(test_mesh)
    phi_init = (x - x_shift) * (x - x_shift) + (y - 0.5) * (y - 0.5) - radius * radius
    phi0 = fd.Function(DGp).interpolate(phi_init)

    phin = solver.solve(phi0, total_t=0.4)
    phi_solution = fd.interpolate(
        sqrt((x - x_shift) * (x - x_shift) + (y - 0.5) * (y - 0.5)) - radius, DGp
    )
    error_numeri = fd.errornorm(phin, phi_solution)
    assert pytest.approx(error, 1e-4) == error_numeri


@pytest.mark.parametrize(
    "test_mesh,x_shift,error, p",
    [
        (fd.UnitSquareMesh(N, N, diagonal="right"), 0.0, 0.051679991911252984, 0),
    ],
)
def test_monitor(test_mesh, x_shift, error, p):

    DGp = fd.FunctionSpace(test_mesh, "DG", p)

    phi_pvd = fd.File("reinit.pvd", target_continuity=fd.H1)
    phi_sol = fd.Function(DGp, name="solution")

    phi_prev = fd.Function(DGp)

    def monitor(ts, i, t, x):
        with phi_sol.dat.vec as v:
            x.copy(v)
        phi_pvd.write(phi_sol)
        print(f"Error: {fd.errornorm(phi_prev, phi_sol)}")
        phi_prev.assign(phi_sol)

    solver_parameters = {
        "ts_type": "rk",
        "ts_view": None,
        "ts_rk_type": "5dp",
        "ts_atol": 1e-5,
        "ts_rtol": 1e-5,
        "ts_dt": 1e-3,
        "ts_monitor": None,
        "ts_adapt_type": "dsp",
        "ts_exact_final_time": "matchstep",
        "h_factor": 5,
    }
    solver = ReinitializationSolver(
        DGp,
        monitor_callback=monitor,
        solver_parameters=solver_parameters,
        poststep=False,
    )

    radius = 0.2
    x, y = fd.SpatialCoordinate(test_mesh)
    phi_init = (x - x_shift) * (x - x_shift) + (y - 0.5) * (y - 0.5) - radius * radius
    phi0 = fd.Function(DGp).interpolate(phi_init)

    phin = solver.solve(phi0, 0.4)
    phi_solution = fd.interpolate(
        sqrt((x - x_shift) * (x - x_shift) + (y - 0.5) * (y - 0.5)) - radius, DGp
    )
    error_numeri = fd.errornorm(phin, phi_solution)
    assert pytest.approx(error, 1e-4) == error_numeri


def test_max_dim():
    mesh = fd.UnitSquareMesh(100, 100)

    assert pytest.approx(max_mesh_dimension(mesh), 1.0)

    mesh = fd.RectangleMesh(10, 20, 5.0, 3.0)

    assert pytest.approx(max_mesh_dimension(mesh), 5.0)

    mesh = fd.CubeMesh(10, 10, 10, 8.0)

    assert pytest.approx(max_mesh_dimension(mesh), 8.0)
