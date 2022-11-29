import firedrake as fd
from firedrake.bcs import DirichletBC
from firedrake.interpolation import interpolate
from firedrake import inner
from ufl.algebra import Abs


import pytest

from letop.optimization import (
    HamiltonJacobiCGSolver,
)

mesh = fd.UnitSquareMesh(50, 50, quadrilateral=True)
x, y = fd.SpatialCoordinate(mesh)
final_t = 0.2

# First case
phi_init_1 = -(x - 1.0) * y
phi_exact_1 = (-(x + final_t - 1.0) * (y - final_t)) * (x < 1.0 - final_t) * (
    y > final_t
)
# Second case
phi_init_2 = x * y
phi_exact_2 = ((x - final_t) * y) * (x > final_t)

# Third case
phi_init_3 = -(x - 1.0) * y
phi_exact_3 = (-(x + final_t - 1.0) * y) * (x < 1.0 - final_t)

# Fourth case
phi_init_4 = -(x) * (y - 1)
phi_exact_4 = (-(x - final_t) * (y + final_t - 1.0)) * (x > final_t) * (
    y < 1.0 - final_t
)


@pytest.mark.parametrize(
    "phi_init, phi_exact, bc_tuple, velocity",
    [
        (
            phi_init_1,
            phi_exact_1,
            (fd.Constant(0.0), (2, 3)),
            fd.Constant((-1.0, 1.0)),
        ),
        (
           phi_init_2,
           phi_exact_2,
           (fd.Constant(0.0), (1)),
           fd.Constant((1.0, 0.0)),
        ),
        (
           phi_init_3,
           phi_exact_3,
           (fd.Constant(0.0), (2)),
           fd.Constant((-1.0, 0.0)),
        ),
        (
           phi_init_4,
           phi_exact_4,
           (fd.Constant(0.0), (1, 4)),
           fd.Constant((1.0, -1.0)),
        ),
    ],
)
def test_cg_hj_solver(phi_init, phi_exact, bc_tuple, velocity):
    V = fd.FunctionSpace(mesh, "CG", 1)
    fd.File('init.pvd').write(fd.interpolate(phi_init, V))
    bc = fd.DirichletBC(V, *bc_tuple)

    phi0 = interpolate(phi_init, V)
    solver_parameters = {
        "ts_atol": 1e-6,
        "ts_rtol": 1e-6,
        "ts_dt": 1e-4,
        "ts_exact_final_time": "matchstep",
        "ts_monitor": None,
        "ts_type": "rosw",
        "ts_rows_type": "2m",
        "ts_adapt_type": "dsp",
        "ts_exact_final_time": "matchstep",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    solver = HamiltonJacobiCGSolver(
        V,
        velocity,
        phi0,
        t_end=final_t,
        bcs=bc,
        solver_parameters=solver_parameters,
    )

    solver.solve()
    error = fd.errornorm(interpolate(phi_exact, V), phi0)
    assert error < 1e-3
    print(f"Error: {error}")
