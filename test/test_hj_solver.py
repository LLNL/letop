import firedrake as fd
from firedrake.bcs import DirichletBC
from firedrake.interpolation import interpolate
from firedrake import inner
from ufl.algebra import Abs


import pytest

from lestofire.optimization import HamiltonJacobiProblem, HamiltonJacobiSolver

mesh = fd.UnitSquareMesh(50, 50, quadrilateral=True)
V = fd.FunctionSpace(mesh, "DG", 1)
x, y = fd.SpatialCoordinate(mesh)
final_t = 0.2

# First case
phi_init_1 = -(x - 1.0) * y + 50.0
phi_exact_1 = interpolate(
    (-(x + final_t - 1.0) * (y - final_t)) * (x < 1.0 - final_t) * (y > final_t) + 50.0,
    V,
)
# Second case
phi_init_2 = x * y + 100.0
phi_exact_2 = interpolate(
    ((x - final_t) * y) * (x > final_t) + 100.0,
    V,
)
# Third case
phi_init_3 = -(x - 1.0) * y
phi_exact_3 = interpolate(
    (-(x + final_t - 1.0) * y) * (x < 1.0 - final_t),
    V,
)
# Fourth case
phi_init_4 = -(x) * (y - 1) + 50.0
phi_exact_4 = interpolate(
    (-(x - final_t) * (y + final_t - 1.0)) * (x > final_t) * (y < 1.0 - final_t) + 50.0,
    V,
)


@pytest.mark.parametrize(
    "phi_init, phi_exact, bc, velocity",
    [
        (
            phi_init_1,
            phi_exact_1,
            DirichletBC(V, fd.Constant(50.0), (2, 3)),
            fd.Constant((-1.0, 1.0)),
        ),
        (
            phi_init_2,
            phi_exact_2,
            DirichletBC(V, fd.Constant(100.0), (1)),
            fd.Constant((1.0, 0.0)),
        ),
        (
            phi_init_3,
            phi_exact_3,
            DirichletBC(V, fd.Constant(0.0), (2)),
            fd.Constant((-1.0, 0.0)),
        ),
        (
            phi_init_4,
            phi_exact_4,
            DirichletBC(V, fd.Constant(50.0), (1, 4)),
            fd.Constant((1.0, -1.0)),
        ),
    ],
)
def test_solve_hj(phi_init, phi_exact, bc, velocity):
    tspan = [0, final_t]

    phi_pvd = fd.File("solution.pvd")
    phi_sol = fd.Function(V, name="solution")

    def monitor(ts, i, t, x):
        with phi_sol.dat.vec as v:
            x.copy(v)
        phi_pvd.write(phi_sol)

    phi0 = interpolate(phi_init, V)

    def H(p):
        return inner(velocity, p)

    def dHdp(p):
        return Abs(velocity)

    problem = HamiltonJacobiProblem(V, phi0, H, dHdp, tspan, bcs=[bc])
    solver_parameters = {
        "ts_type": "rk",
        "ts_rk_type": "5dp",
        "ts_view": None,
        "ts_atol": 1e-6,
        "ts_rtol": 1e-6,
        "ts_dt": 1e-4,
        "ts_monitor": None,
        "ts_exact_final_time": "matchstep",
        "ts_adapt_type": "dsp",
    }
    solver = HamiltonJacobiSolver(
        problem, monitor_callback=monitor, solver_parameters=solver_parameters
    )
    solver.solve()
    error = fd.errornorm(phi_exact, phi0)
    assert error < 1e-3
    print(f"Error: {error}")


def test_solve_hj_restart():
    phi_init = -(x - 1.0) * y + 50.0
    bc = DirichletBC(V, fd.Constant(50.0), (2, 3))
    velocity = fd.Constant((-1.0, 1.0))
    tspan = [0, final_t]

    phi_pvd = fd.File("solution.pvd")
    phi_sol = fd.Function(V, name="solution")

    def monitor(ts, i, t, x):
        with phi_sol.dat.vec as v:
            x.copy(v)
        phi_pvd.write(phi_sol)

    phi0 = interpolate(phi_init, V)

    def H(p):
        return inner(velocity, p)

    def dHdp(p):
        return Abs(velocity)

    problem = HamiltonJacobiProblem(V, phi0, H, dHdp, tspan, bcs=[bc])
    solver_parameters = {
        "ts_type": "rk",
        "ts_rk_type": "5dp",
        "ts_view": None,
        "ts_atol": 1e-6,
        "ts_rtol": 1e-6,
        "ts_dt": 1e-4,
        "ts_monitor": None,
        "ts_exact_final_time": "matchstep",
        "ts_adapt_type": "dsp",
    }
    solver = HamiltonJacobiSolver(
        problem, monitor_callback=monitor, solver_parameters=solver_parameters
    )
    phi_new = solver.solve()
    solver.ts.setMaxTime(0.4)
    solver.solve(phi_new)
    phi_exact = interpolate(
        (-(x + 3.0 * final_t - 1.0) * (y - 3.0 * final_t))
        * (x < 1.0 - 3.0 * final_t)
        * (y > 3.0 * final_t)
        + 50.0,
        V,
    )
    error = fd.errornorm(phi_exact, phi0)
    print(f"Error: {error}")
    assert error < 1e-3


def test_hj_no_bc():
    from firedrake import inner, grad, dx

    tspan = [0, 1]

    VelSpace = fd.VectorFunctionSpace(mesh, "CG", 1)
    u, v = fd.TrialFunction(VelSpace), fd.TestFunction(VelSpace)
    gamma = fd.Constant(2.0)
    b = gamma * inner(grad(u), grad(v)) * dx + inner(u, v) * dx
    x, y = fd.SpatialCoordinate(mesh)
    sole_field = interpolate(fd.as_vector([y - 0.5, -(x - 0.5)]), VelSpace)
    L = inner(sole_field, v) * dx
    bc1 = DirichletBC(VelSpace.sub(0), fd.Constant(0.0), (1, 2))
    bc2 = DirichletBC(VelSpace.sub(1), fd.Constant(0.0), (3, 4))

    direct_parameters = {
        "snes_type": "ksponly",
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    th_temp = fd.Function(VelSpace)
    fd.solve(b == L, th_temp, bcs=[bc1, bc2], solver_parameters=direct_parameters)
    Vvec = fd.VectorFunctionSpace(mesh, "DG", V.ufl_element().degree())
    th = fd.Function(Vvec)
    th.interpolate(th_temp)
    fd.File("solenoidal.pvd").write(th)

    phi_pvd = fd.File("solution.pvd")
    phi_sol = fd.Function(V, name="solution")

    def monitor(ts, i, t, x):
        with phi_sol.dat.vec as v:
            x.copy(v)
        phi_pvd.write(phi_sol)

    def H(p):
        return inner(th, p)

    def dHdp(p):
        return Abs(th)

    phi_init = x * y
    phi0 = interpolate(phi_init, V)
    phi0.rename("solution")
    problem = HamiltonJacobiProblem(V, phi0, H, dHdp, tspan)
    solver_parameters = {
        "ts_type": "rk",
        "ts_view": None,
        "ts_rk_type": "5dp",
        "ts_atol": 1e-6,
        "ts_rtol": 1e-6,
        # "ts_dt": 1e-8,
        "ts_max_time": 100.0,
        "ts_monitor": None,
        "ts_adapt_type": "dsp",
    }
    solver = HamiltonJacobiSolver(
        problem, monitor_callback=monitor, solver_parameters=solver_parameters
    )
    solver.solve()

    import numpy as np

    np.isclose(fd.norm(phi0), 0.33306171841130205)


if __name__ == "__main__":
    test_solve_hj(
        phi_init_1,
        phi_exact_1,
        DirichletBC(V, fd.Constant(50.0), (2, 3)),
        fd.Constant((-1.0, 1.0)),
    )
