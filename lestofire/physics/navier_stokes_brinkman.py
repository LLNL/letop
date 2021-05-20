from cmath import tau
import firedrake as fd
from firedrake import inner, dot, grad, div, dx, ds, dot, dS, jump, avg, Constant, exp
from firedrake.function import Function
from pyadjoint.enlisting import Enlist
import ufl
from .utils import hs
from typing import Callable, Union
from ufl.algebra import Product
from functools import partial


def NavierStokesBrinkmannForm(
    W: fd.FunctionSpace,
    w: fd.Function,
    phi: Union[fd.Function, Product],
    nu,
    brinkmann_penalty=None,
    brinkmann_min=0.0,
    design_domain=None,
    no_flow_domain=None,
    hs: Callable = hs,
) -> ufl.form:
    """Returns the Galerkin Least Squares formulation for the Navier-Stokes problem with a Brinkmann term

    Args:
        W (fd.FunctionSpace): [description]
        w (fd.Function): [description]
        phi (fd.Function): [description]
        nu ([type]): [description]
        brinkmann_penalty ([type], optional): [description]. Defaults to None.
        design_domain ([type], optional): Region where the level set is defined. Defaults to None.
        no_flow_domain ([type], optional): Region where there is no flow and the Brinkmann term is imposed. Defaults to None.

    Returns:
        ufl.form: Nonlinear form
    """
    mesh = phi.ufl_domain()

    W_elem = W.ufl_element()
    assert isinstance(W_elem, fd.MixedElement)
    assert W_elem.num_sub_elements() == 2

    for W_sub_elem in W_elem.sub_elements():
        assert W_sub_elem.family() == "Lagrange"
        assert W_sub_elem.degree() == 1
    assert isinstance(W_elem.sub_elements()[0], fd.VectorElement)

    v, q = fd.TestFunctions(W)
    u, p = fd.split(w)

    # Main NS form
    F = (
        nu * inner(grad(u), grad(v)) * dx
        + inner(dot(grad(u), u), v) * dx
        - p * div(v) * dx
        + div(u) * q * dx
    )

    # Brinkmann terms for design
    def add_measures(list_dd, **kwargs):
        return sum([dx(dd, kwargs) for dd in list_dd[1::]], dx(list_dd[0]))

    def alpha(phi):
        return Constant(brinkmann_penalty) * hs(phi) + Constant(brinkmann_min)

    if brinkmann_penalty:
        if design_domain is not None:
            dx_brinkmann = add_measures(Enlist(design_domain))
        else:
            dx_brinkmann = dx

        F = F + alpha(phi) * inner(u, v) * dx_brinkmann

    if no_flow_domain:
        dx_no_flow = partial(add_measures, Enlist(no_flow_domain))
        F = F + Constant(brinkmann_penalty) * inner(u, v) * dx_no_flow()

    # GLS stabilization
    R_U = dot(u, grad(u)) - nu * div(grad(u)) + grad(p)
    beta_gls = 0.9
    h = fd.CellSize(mesh)
    tau_gls = beta_gls * (
        (4.0 * dot(u, u) / h ** 2) + 9.0 * (4.0 * nu / h ** 2) ** 2
    ) ** (-0.5)
    degree = 8

    theta_U = dot(u, grad(v)) - nu * div(grad(v)) + grad(q)
    F = F + tau_gls * inner(R_U, theta_U) * dx(degree=degree)

    if brinkmann_penalty:
        tau_gls_alpha = beta_gls * (
            (4.0 * dot(u, u) / h ** 2)
            + 9.0 * (4.0 * nu / h ** 2) ** 2
            + (alpha(phi) / 1.0) ** 2
        ) ** (-0.5)
        R_U_alpha = R_U + alpha(phi) * u
        theta_alpha = theta_U + alpha(phi) * v

        F = F + tau_gls_alpha * inner(R_U_alpha, theta_alpha) * dx_brinkmann(
            degree=degree
        )
        if (
            design_domain is not None
        ):  # Substract this domain from the original integral
            F = F - tau_gls * inner(R_U, theta_U) * dx_brinkmann(degree=degree)

    if no_flow_domain:
        tau_gls_alpha = beta_gls * (
            (4.0 * dot(u, u) / h ** 2)
            + 9.0 * (4.0 * nu / h ** 2) ** 2
            + (Constant(brinkmann_penalty) / 1.0) ** 2
        ) ** (-0.5)
        R_U_alpha = R_U + Constant(brinkmann_penalty) * u
        theta_alpha = theta_U + Constant(brinkmann_penalty) * v

        F = F + tau_gls_alpha * inner(R_U_alpha, theta_alpha) * dx_no_flow(
            degree=degree
        )
        # Substract this domain from the original integral
        F = F - tau_gls * inner(R_U, theta_U) * dx_no_flow(degree=degree)
    return F


class NavierStokesBrinkmannSolver(object):
    def __init__(self, problem: fd.NonlinearVariationalProblem, **kwargs) -> None:
        """Same than NonlinearVariationalSolver, but with just the SIMPLE preconditioner by default
        Args:
            problem ([type]): [description]
            nullspace ([type], optional): [description]. Defaults to None.
            solver_parameters ([type], optional): [description]. Defaults to None.
        """
        solver_parameters_default = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "l2",
            "snes_linesearch_maxstep": 1.0,
            # "snes_monitor": None,
            # "snes_linesearch_monitor": None,
            "snes_rtol": 1.0e-5,
            "snes_atol": 1.0e-8,
            "snes_stol": 0.0,
            "snes_max_linear_solve_fail": 10,
            "snes_converged_reason": None,
            "ksp_type": "fgmres",
            "mat_type": "aij",
            # "default_sub_matrix_type": "aij",
            "ksp_rtol": 1.0e-5,
            "ksp_atol": 1.0e-8,
            "ksp_max_it": 2000,
            # "ksp_monitor": None,
            "ksp_converged_reason": None,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_factorization_type": "full",
            "pc_fieldsplit_schur_precondition": "selfp",
            "fieldsplit_0": {
                "ksp_type": "richardson",
                "ksp_richardson_self_scale": False,
                # "ksp_converged_reason": None,
                "ksp_max_it": 1,
                # "ksp_monitor": None,
                "pc_type": "ml",
                "ksp_atol": 1e-2,
                "pc_mg_cycle_type": "v",
                "pc_mg_type": "full",
            },
            "fieldsplit_1": {
                # "ksp_monitor": None,
                # "ksp_converged_reason": None,
                "ksp_type": "preonly",
                "pc_type": "ml",
            },
            "fieldsplit_1_upper_ksp_type": "preonly",
            "fieldsplit_1_upper_pc_type": "jacobi",
        }
        solver_parameters = kwargs.pop("solver_parameters", None)
        if solver_parameters:
            solver_parameters_default.update(solver_parameters)

        self.solver = fd.NonlinearVariationalSolver(
            problem,
            solver_parameters=solver_parameters_default,
            **kwargs
        )

    def solve(self):
        self.solver.solve()
