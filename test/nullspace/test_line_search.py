import firedrake as fd
import firedrake_adjoint as fda
import numpy as np
from firedrake import dx
from letop.optimization.nullspace_shape import line_search
from letop.optimization import InfDimProblem
from letop.levelset import LevelSetFunctional, RegularizationSolver
from letop.physics import hs
from numpy.testing import assert_allclose


def merit_eval_new(AJ, J, AC, C):
    return J


def test_line_search():
    mesh = fd.UnitSquareMesh(50, 50)

    # Shape derivative
    S = fd.VectorFunctionSpace(mesh, "CG", 1)
    s = fd.Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    # Level set
    PHI = fd.FunctionSpace(mesh, "CG", 1)
    x, y = fd.SpatialCoordinate(mesh)

    with fda.stop_annotating():
        phi = fd.interpolate(-(x - 0.5), PHI)

    solver_parameters = {
        "ts_atol": 1e-4,
        "ts_rtol": 1e-4,
        "ts_dt": 1e-2,
        "ts_exact_final_time": "matchstep",
        "ts_monitor": None,
    }

    ## Search direction
    with fda.stop_annotating():
        delta_x = fd.interpolate(fd.as_vector([-100 * x, 0.0]), S)
        delta_x.rename("velocity")

    # Cost function
    J = fd.assemble(hs(-phi) * dx)

    # Reduced Functional
    c = fda.Control(s)
    Jhat = LevelSetFunctional(J, c, phi)

    # InfDim Problem
    beta_param = 0.08
    reg_solver = RegularizationSolver(
        S, mesh, beta=beta_param, gamma=1e5, dx=dx
    )
    solver_parameters = {"hj_solver": solver_parameters}
    problem = InfDimProblem(
        Jhat, reg_solver, solver_parameters=solver_parameters
    )

    with fda.stop_annotating():
        problem.delta_x.assign(delta_x)

    AJ, AC = 1.0, 1.0
    C = np.array([])
    merit = merit_eval_new(AJ, J, AC, C)

    rtol = 1e-4
    newJ, newG, newH = line_search(
        problem,
        merit_eval_new,
        merit,
        AJ,
        AC,
        dt=1.0,
        tol_merit=rtol,
        maxtrials=20,
    )

    assert_allclose(newJ, J, rtol=rtol)


if __name__ == "__main__":
    test_line_search()
