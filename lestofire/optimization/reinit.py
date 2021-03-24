from firedrake import (
    VectorFunctionSpace,
    TrialFunction,
    TestFunction,
    FacetNormal,
    sqrt,
    Constant,
    dx,
    dS,
    ds,
    Max,
    as_vector,
    inner,
    lhs,
    rhs,
    div,
    jump,
    Function,
    solve,
    File,
    H1,
)
from firedrake.bcs import DirichletBC
from firedrake.norms import errornorm
from firedrake.ufl_expr import CellSize
from pyadjoint.tape import no_annotations
from ufl.geometry import CellDiameter

from lestofire.utils import petsc_print


direct_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    #    "pc_factor_mat_solver_type": "mumps",
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
    def __init__(self, mesh, dt=1e-4, n_steps=10, iterative=False, h_factor=2.0):
        self.dt = dt
        self.n_steps = n_steps
        self.phi_solution = None
        self.phi_pvd = File("reinit.pvd",  target_continuity=H1)
        self.h_factor = h_factor

        if iterative:
            self.parameters = iterative_parameters
        else:
            self.parameters = direct_parameters

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
        VDG0 = VectorFunctionSpace(mesh, "DG", 0)
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
        #a1 -= inner(v, diag(phi0 * clip(-n)) * n) * ds
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
        #a2 -= inner(v, diag(phi0 * clip(n)) * n) * ds

        direct_parameters = {
            "snes_type": "ksponly",
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        #Delta_x = 10 / 50
        #Delta_x = CellDiameter(mesh) * 20.0
        #Delta_x = CellDiameter(mesh) * 5.0
        #Delta_x = CellDiameter(mesh) * 5.0
        Delta_x = CellDiameter(mesh) * self.h_factor

        def sign(phi, phi_x, phi_y):
            return phi / sqrt(phi * phi + Delta_x * Delta_x * (phi_x ** 2 + phi_y ** 2))

        phi_signed = phi0.copy(deepcopy=True)


        jacobi_solver = {"ksp_type": "preonly", "pc_type": "jacobi"}

        p1_0 = Function(VDG0)
        p2_0 = Function(VDG0)
        solve(lhs(a1) == rhs(a1), p1_0, solver_parameters=jacobi_solver)
        solve(lhs(a2) == rhs(a2), p2_0, solver_parameters=jacobi_solver)

        def H(p):
            return sign(phi_signed, (p1_0[0] + p2_0[0])/ Constant(2.0), (p1_0[1] + p2_0[1])/ Constant(2.0)) * (sqrt(inner(p, p)) - 1)

        def dHdp(p_x, p_y):
            return as_vector(
                [
                    abs(sign(phi_signed, (p1_0[0] + p2_0[0])/ Constant(2.0), (p1_0[1] + p2_0[1])/ Constant(2.0)) * p_x / sqrt(p_x * p_x + p_y * p_y + 1e-7)),
                    abs(sign(phi_signed, (p1_0[0] + p2_0[0])/ Constant(2.0), (p1_0[1] + p2_0[1])/ Constant(2.0)) * p_y / sqrt(p_x * p_x + p_y * p_y + 1e-7)),
                ]
            )

        def alpha(p1_x, p2_x, p1_y, p2_y):
            return Max(dHdp(p1_x, p1_y)[0], dHdp(p2_x, p2_y)[0])

        def beta(p1_x, p2_x, p1_y, p2_y):
            return Max(dHdp(p1_x, p1_y)[1], dHdp(p2_x, p2_y)[1])

        p1 = Function(VDG0)
        p2 = Function(VDG0)
        phin = Function(DG0)
        for j in range(self.n_steps):
            solve(lhs(a1) == rhs(a1), p1, solver_parameters=jacobi_solver)
            solve(lhs(a2) == rhs(a2), p2, solver_parameters=jacobi_solver)

            if j % 1 == 0:
                if self.phi_pvd:
                    self.phi_pvd.write(phi0)

            dt = self.dt
            b = (phi - phi0) * rho / Constant(dt) * dx + (
                H((p1 + p2) / Constant(2.0))
                - Constant(1.0 / 2.0)
                * alpha(p1[0], p2[0], p1[1], p2[1])
                * (p1[0] - p2[0])
                - Constant(1.0 / 2.0)
                * beta(p1[0], p2[0], p1[1], p2[1])
                * (p1[1] - p2[1])
            ) * rho * dx
            solve(lhs(b) == rhs(b), phin, solver_parameters=direct_parameters)
            print(f"Residual: {errornorm(phi0, phin)}")
            phi0.assign(phin)

        return phi0