import firedrake as fd
from firedrake import sqrt
from lestofire.optimization import ReinitializationSolver
import pytest


N = 50


@pytest.mark.parametrize(
    "test_mesh,x_shift,error, p",
    [
        (fd.UnitSquareMesh(N, N, diagonal="right"), 0.0, 0.051326588312199745, 0),
        (fd.Mesh("./unstructured_rectangle.msh"), 0.0, 0.05447032030474453, 0),
        (fd.UnitSquareMesh(N, N, diagonal="right"), 0.5, 0.02172097234980319, 0),
        (fd.Mesh("./unstructured_rectangle.msh"), 0.5, 0.018930025053364902, 0),
        (fd.UnitSquareMesh(N, N, diagonal="right"), 0.0, 0.056360176396345746, 1),
        (fd.Mesh("./unstructured_rectangle.msh"), 0.0, 0.05428718022564955, 1),
        (fd.UnitSquareMesh(N, N, diagonal="right"), 0.5, 0.023154545651903394, 1),
        (fd.Mesh("./unstructured_rectangle.msh"), 0.5, 0.018158858201365733, 1),
        (fd.UnitSquareMesh(12, 12, quadrilateral=True), 0.5, 0.055021199124857104, 2),
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
    }
    solver = ReinitializationSolver(
        DGp,
        5.0,
        monitor_callback=monitor,
        stopping_criteria=0.0,
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
        (fd.UnitSquareMesh(N, N, diagonal="right"), 0.0, 0.051326588312199745, 0),
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
    }
    solver = ReinitializationSolver(
        DGp,
        5.0,
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
