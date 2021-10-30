import firedrake as fd
import firedrake_ts as fdts
from firedrake import inner, dot, grad, div, dx, ds, dS, jump, avg
from letop.physics import AdvectionDiffusionGLS, InteriorBC
from typing import Union, List, Callable


def HamiltonJacobiCGSolver(
    V: fd.FunctionSpace,
    theta: fd.Function,
    phi: fd.Function,
    t_end: float = 5000.0,
    bcs: Union[fd.DirichletBC, List[fd.DirichletBC]] = None,
    monitor: Callable = None,
    solver_parameters=None,
    pre_jacobian_callback=None,
    post_jacobian_callback=None,
    pre_function_callback=None,
    post_function_callback=None,
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
        :kwarg pre_jacobian_callback: A user-defined function that will
               be called immediately before Jacobian assembly. This can
               be used, for example, to update a coefficient function
               that has a complicated dependence on the unknown solution.
        :kwarg pre_function_callback: As above, but called immediately
               before residual assembly.
        :kwarg post_jacobian_callback: As above, but called after the
               Jacobian has been assembled.
        :kwarg post_function_callback: As above, but called immediately
               after residual assembly.


    Returns:
        fdts.DAESolver: DAESolver configured to solve the advection-diffusion equation
    """

    default_peclet_inv = 1e-4
    if solver_parameters:
        PeInv = solver_parameters.get("peclet_number", default_peclet_inv)
    else:
        PeInv = default_peclet_inv

    phi_t = fd.Function(V)
    # Galerkin residual
    F = AdvectionDiffusionGLS(V, theta, phi, phi_t, PeInv=PeInv)

    problem = fdts.DAEProblem(F, phi, phi_t, (0.0, t_end), bcs=bcs)
    parameters = {
        "ts_type": "rosw",
        "ts_rows_type": "2m",
        "ts_adapt_type": "dsp",
        "ts_exact_final_time": "matchstep",
        "ts_atol": 1e-7,
        "ts_rtol": 1e-7,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    if solver_parameters:
        parameters.update(solver_parameters)

    return fdts.DAESolver(
        problem,
        solver_parameters=parameters,
        monitor_callback=monitor,
        pre_function_callback=pre_function_callback,
        post_function_callback=post_function_callback,
        pre_jacobian_callback=pre_jacobian_callback,
        post_jacobian_callback=post_jacobian_callback,
    )
