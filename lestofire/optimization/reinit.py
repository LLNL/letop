from firedrake import (
    H1,
    Constant,
    FacetNormal,
    File,
    Function,
    FunctionSpace,
    Max,
    TestFunction,
    TrialFunction,
    as_vector,
    div,
    dS,
    ds,
    dx,
    inner,
    jump,
    lhs,
    rhs,
    solve,
    sqrt,
    LinearVariationalProblem,
    LinearVariationalSolver,
)
from firedrake.bcs import DirichletBC
from firedrake.norms import errornorm
from firedrake.ufl_expr import CellSize
from lestofire.utils import petsc_print
from pyadjoint.tape import no_annotations
from ufl import VectorElement
from ufl.algebra import Abs
from ufl.geometry import CellDiameter
from pyop2.profiling import timed_function

direct_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

iterative_parameters = {
    "ksp_type": "cg",
    "ksp_max_it": 2000,
    "ksp_atol": 1e-9,
    "ksp_rtol": 1e-9,
    "pc_type": "bjacobi",
    "ksp_converged_reason": None,
}


def max_component(vector1, vector2):
    """UFL implementation of max component wise between two vectors

    Args:
        vector1 ([type]): [description]
        vector2 ([type]): [description]

    Returns:
        [type]: UFL expression with the component wise max
    """
    assert vector1.ufl_shape == vector2.ufl_shape and len(vector1.ufl_shape) == 1
    return as_vector([Max(vector1[i], vector2[i]) for i in range(vector1.ufl_shape[0])])


class ReinitSolverDG(object):
    def __init__(
        self, mesh, dt=1e-4, n_steps=10, iterative=False, h_factor=2.0, phi_pvd=None
    ):
        self.dt = dt
        self.n_steps = n_steps
        self.phi_solution = None
        self.phi_pvd = phi_pvd
        self.h_factor = h_factor

        if iterative:
            self.parameters = iterative_parameters
        else:
            self.parameters = direct_parameters

    @timed_function("Solve Reinitialization solver")
    @no_annotations
    def solve(self, phi0):
        """Jue Yan, Stanley Osher,
        A local discontinuous Galerkin method for directly solving Hamilton–Jacobi equations,
        Journal of Computational Physics,
        Volume 230, Issue 1,
        2011,
        Pages 232-244,

        Args:
            phi0 ([type]): Initial level set
        Returns:
            [type]: Solution after "n_steps" number of steps
        """
        from functools import partial

        import numpy as np

        DG0 = phi0.function_space()
        mesh = DG0.ufl_domain()
        n = FacetNormal(mesh)
        phi = TrialFunction(DG0)
        rho = TestFunction(DG0)
        DG_elem = DG0.ufl_element()
        dim = mesh.geometric_dimension()
        VDG_elem = VectorElement(
            DG_elem.family(), DG_elem.cell(), DG_elem.degree(), dim
        )
        VDG0 = FunctionSpace(mesh, VDG_elem)
        p = TrialFunction(VDG0)
        v = TestFunction(VDG0)

        def clip(vector):
            from ufl import conditional, ge

            return as_vector([conditional(ge(vi, 0.0), 1.0, 0.0) for vi in vector])

        from ufl import diag

        a1 = (
            inner(p, v) * dx
            + phi0 * div(v) * dx
            - inner(
                v("+"),
                diag(phi0("-") * clip(n("+")) + phi0("+") * clip(n("-"))) * n("+"),
            )
            * dS
            - inner(
                v("-"),
                diag(phi0("-") * clip(n("+")) + phi0("+") * clip(n("-"))) * n("-"),
            )
            * dS
        )
        a1 -= inner(v, n) * phi0 * ds
        # a1 -= inner(v, diag(phi0 * clip(-n)) * n) * ds
        a2 = (
            inner(p, v) * dx
            + phi0 * div(v) * dx
            - inner(
                v("+"),
                diag(phi0("+") * clip(n("+")) + phi0("-") * clip(n("-"))) * n("+"),
            )
            * dS
            - inner(
                v("-"),
                diag(phi0("+") * clip(n("+")) + phi0("-") * clip(n("-"))) * n("-"),
            )
            * dS
        )
        a2 -= inner(v, n) * phi0 * ds
        # a2 -= inner(v, diag(phi0 * clip(n)) * n) * ds

        # Delta_x = 10 / 50
        # Delta_x = CellDiameter(mesh) * 20.0
        # Delta_x = CellDiameter(mesh) * 5.0
        # Delta_x = CellDiameter(mesh) * 5.0
        Delta_x = CellDiameter(mesh) * self.h_factor

        def sign(phi, grad_phi):
            return phi / sqrt(
                phi * phi + Delta_x * Delta_x * (inner(grad_phi, grad_phi))
            )

        phi_signed = phi0.copy(deepcopy=True)

        jacobi_solver = {
            "ksp_type": "preonly",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }

        p1_0 = Function(VDG0)
        p2_0 = Function(VDG0)
        solve(lhs(a1) == rhs(a1), p1_0, solver_parameters=jacobi_solver)
        solve(lhs(a2) == rhs(a2), p2_0, solver_parameters=jacobi_solver)

        grad_phi_0 = (p1_0 + p2_0) / Constant(2.0)

        def H(p):
            return sign(phi_signed, grad_phi_0) * (sqrt(inner(p, p)) - 1)

        def dHdp(p):
            return Abs(sign(phi_signed, grad_phi_0) * p / sqrt(inner(p, p) + 1e-7))

        def alpha(p1, p2):
            return as_vector(
                [
                    Max(dHdp_p1_i, dHdp_p2_i)
                    for dHdp_p1_i, dHdp_p2_i in zip(dHdp(p1), dHdp(p2))
                ]
            )

        p1 = Function(VDG0)
        p2 = Function(VDG0)
        phin = Function(DG0)

        problem_phi_1 = LinearVariationalProblem(lhs(a1), rhs(a1), p1)
        solver_phi_1 = LinearVariationalSolver(
            problem_phi_1, solver_parameters=jacobi_solver
        )

        problem_phi_2 = LinearVariationalProblem(lhs(a2), rhs(a2), p2)
        solver_phi_2 = LinearVariationalSolver(
            problem_phi_2, solver_parameters=jacobi_solver
        )
        dt = self.dt
        b = (phi - phi0) * rho / Constant(dt) * dx + (
            H((p1 + p2) / Constant(2.0))
            - Constant(1.0 / 2.0) * inner(alpha(p1, p2), (p1 - p2))
        ) * rho * dx

        problem_phi0 = LinearVariationalProblem(lhs(b), rhs(b), phin)
        solver_phi0 = LinearVariationalSolver(
            problem_phi0, solver_parameters=jacobi_solver
        )
        for j in range(self.n_steps):
            solver_phi_1.solve()
            solver_phi_2.solve()

            # if j % 1 == 0:
            if self.phi_pvd:
                self.phi_pvd.write(phi0)

            solver_phi0.solve()
            print(f"Residual: {errornorm(phi0, phin)}")
            phi0.assign(phin)

        return phi0
