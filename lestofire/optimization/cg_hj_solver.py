import firedrake as fd
import firedrake_ts as fdts
from firedrake import inner, dot, grad, div, dx, ds, dS, jump, avg
from lestofire.physics import AdvectionDiffusionSUPG
from typing import Union, List, Callable, Tuple


def HamiltonJacobiCGSolver(
    V: fd.FunctionSpace,
    theta: fd.Function,
    phi: fd.Function,
    t_end: float = 5000.0,
    bcs: Union[fd.DirichletBC, List[fd.DirichletBC]] = None,
    monitor: Callable = None,
    solver_parameters=None,
) -> fdts.DAESolver:
    """Builds the solver for the advection-diffusion equation (Hamilton-Jacobi in the context of
    topology optimization) which is used to advect the level set)

    Args:
        V (fd.FunctionSpace): Function space of the level set
        theta (fd.Function): Velocity function
        phi (fd.Function): Level set
        t_end (float, optional): Max time of simulation. Defaults to 5000.0.
        bcs (Union[fd.DirichletBC, List[fd.DirichletBC]], optional): BC for the equation. Defaults to None.
        monitor (Callable, optional): Monitor function called each time step. Defaults to None.
        solver_parameters ([type], optional): Solver options. Defaults to None.

    Returns:
        fdts.DAESolver: DAESolver configured to solve the advection-diffusion equation
    """
    phi_t = fd.Function(V)
    # Galerkin residual
    k = fd.Constant(1e-6)
    F = AdvectionDiffusionSUPG(V, theta, k, phi, phi_t)

    problem = fdts.DAEProblem(F, phi, phi_t, (0.0, t_end), bcs=bcs)
    parameters = {
        "ts_monitor": None,
        "ts_type": "rosw",
        "ts_rows_type": "2m",
        "ts_adapt_type": "dsp",
        "ts_exact_final_time": "matchstep",
        "ts_atol": 1e-7,
        "ts_rtol": 1e-7,
        # "snes_monitor" : None,
        # "ksp_monitor" : None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    if solver_parameters:
        parameters.update(solver_parameters)

    return fdts.DAESolver(
        problem, solver_parameters=parameters, monitor_callback=monitor
    )
