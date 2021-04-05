import firedrake as fd
from firedrake import inner, dot, grad, div, dx, ds, dot, dS, jump, avg, Constant, exp
from firedrake.function import Function
from pyadjoint.enlisting import Enlist
import ufl
from .utils import hs


def NavierStokesBrinkmannForm(
    W: fd.FunctionSpace,
    w: fd.Function,
    phi: fd.Function,
    nu,
    brinkmann_penalty=None,
    brinkmann_min=0.0,
    design_domain=None,
    no_flow_domain=None,
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
    def add_measures(list_dd):
        return sum([dx(dd) for dd in list_dd[1::]], dx(list_dd[0]))

    if brinkmann_penalty:
        if design_domain:
            dx_brinkmann = add_measures(Enlist(design_domain))
        else:
            dx_brinkmann = dx
        F = (
            F
            + (Constant(brinkmann_penalty) * hs(phi) + Constant(brinkmann_min))
            * inner(u, v)
            * dx_brinkmann
        )

    if no_flow_domain:
        dx_no_flow = add_measures(Enlist(no_flow_domain))
        F = F + brinkmann_penalty * inner(u, v) * dx_no_flow

    # GLS stabilization
    R_U = dot(u, grad(u)) - nu * div(grad(u)) + grad(p)
    h = fd.CellDiameter(mesh)

    beta_gls = 0.9
    tau_gls = beta_gls * (
        (4.0 * dot(u, u) / h ** 2) + 9.0 * (4.0 * nu / h ** 2) ** 2
    ) ** (-0.5)

    F = F + tau_gls * inner(R_U, dot(u, grad(v)) - nu * div(grad(v)) + grad(q)) * dx(
        degree=4
    )
    return F


class NavierStokesBrinkmannSolver(object):
    def __init__(
        self,
        problem: fd.NonlinearVariationalProblem,
        nullspace=None,
        solver_parameters=None,
    ) -> None:
        """Same than NonlinearVariationalSolver, but with just the SIMPLE preconditioner by default

        Args:
            problem ([type]): [description]
            nullspace ([type], optional): [description]. Defaults to None.
            solver_parameters ([type], optional): [description]. Defaults to None.
        """
        solver_parameters_simple = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "l2",
            "snes_linesearch_maxstep": 1.0,
            "snes_monitor": None,
            "snes_linesearch_monitor": None,
            "snes_rtol": 1.0e-4,
            "snes_atol": 1.0e-8,
            "snes_stol": 0.0,
            "snes_max_linear_solve_fail": 10,
            "snes_converged_reason": None,
            "ksp_type": "fgmres",
            "mat_type": "aij",
            # "default_sub_matrix_type": "aij",
            "ksp_rtol": 1.0e-4,
            "ksp_atol": 1.0e-8,
            "ksp_max_it": 500,
            "ksp_monitor": None,
            "ksp_converged_reason": None,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_factorization_type": "full",
            "pc_fieldsplit_schur_precondition": "selfp",
            "fieldsplit_0": {
                "ksp_type": "richardson",
                "ksp_richardson_self_scale": False,
                "ksp_converged_reason": None,
                "ksp_max_it": 1,
                # "ksp_monitor": None,
                "pc_type": "ml",
                "ksp_atol": 1e-2,
                "pc_mg_cycle_type": "v",
                "pc_mg_type": "full",
            },
            "fieldsplit_1": {
                # "ksp_monitor": None,
                "ksp_converged_reason": None,
                "ksp_type": "preonly",
                "pc_type": "ml",
            },
            "fieldsplit_1_upper_ksp_type": "preonly",
            "fieldsplit_1_upper_pc_type": "jacobi",
        }
        self.solver_parameters = solver_parameters_simple
        if solver_parameters:
            self.solver_parameters.update(solver_parameters)

        self.solver = fd.NonlinearVariationalSolver(
            problem,
            nullspace=nullspace,
            solver_parameters=self.solver_parameters,
        )

    def solve(self):
        self.solver.solve()