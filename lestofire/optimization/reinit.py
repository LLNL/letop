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
from pyadjoint.tape import no_annotations

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
    def __init__(self, mesh, dt=1e-4, n_steps=10, iterative=False):
        self.dt = dt
        self.n_steps = n_steps
        self.phi_solution = None
        self.phi_pvd = File("reinit.pvd", target_continuity=H1)

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
        Deltax = 1.0 / np.square(mesh.num_cells())
        n = FacetNormal(mesh)
        phi = TrialFunction(DG0)
        rho = TestFunction(DG0)
        VDG0 = VectorFunctionSpace(mesh, "DG", 0)
        p = TrialFunction(VDG0)
        v = TestFunction(VDG0)
        cmp_to_zero = partial(max_component, Constant((0.0, 0.0)))

        a1 = (
            inner(p, v) * dx
            + phi0
            * div(v)
            * dx  # Replace this term with phi0 * div(v) for the multdim example
            - inner(
                jump(v),
                phi0("+") * cmp_to_zero(n("+")) + phi0("-") * cmp_to_zero(n("-")),
            )
            * dS
        )
        a1 -= inner(v, n) * phi0 * ds
        a2 = (
            inner(p, v) * dx
            + phi0
            * div(v)
            * dx  # Replace this term with phi0 * div(v) for the multdim example
            - inner(
                jump(v),
                phi0("+") * cmp_to_zero(-n("+")) + phi0("-") * cmp_to_zero(-n("-")),
            )
            * dS
        )
        a2 -= inner(v, n) * phi0 * ds

        p1 = Function(VDG0)
        p2 = Function(VDG0)
        direct_parameters = {
            "snes_type": "ksponly",
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        def sign(phi):
            return phi / sqrt(phi * phi + Deltax * Deltax)

        phi_signed = phi0.copy(deepcopy=True)

        def H(p):
            return sign(phi_signed) * (sqrt(inner(p, p)) - 1)

        def dHdp(p_x, p_y):
            return as_vector(
                [
                    abs(sign(phi_signed) * p_x / sqrt(p_x * p_x + p_y * p_y + 1e-7)),
                    abs(sign(phi_signed) * p_y / sqrt(p_x * p_x + p_y * p_y + 1e-7)),
                ]
            )

        def alpha(p1_x, p2_x, p1_y, p2_y):
            return Max(dHdp(p1_x, p1_y)[0], dHdp(p2_x, p2_y)[0])

        def beta(p1_x, p2_x, p1_y, p2_y):
            return Max(dHdp(p1_x, p1_y)[1], dHdp(p2_x, p2_y)[1])

        jacobi_solver = {"ksp_type": "preonly", "pc_type": "jacobi"}

        for _ in range(self.n_steps):
            solve(lhs(a1) == rhs(a1), p1, solver_parameters=jacobi_solver)
            solve(lhs(a2) == rhs(a2), p2, solver_parameters=jacobi_solver)

            if self.phi_pvd:
                self.phi_pvd.write(phi0)

            dt = self.dt
            b = (phi - phi0) * rho / Constant(dt) * dx + (
                H((p1 + p2) / Constant(2.0))
                - Constant(1.0 / 2.0)
                * alpha(p1[0], p2[0], p1[1], p2[1])
                * (-p1[0] + p2[0])
                - Constant(1.0 / 2.0)
                * beta(p1[0], p2[0], p1[1], p2[1])
                * (-p1[1] + p2[1])
            ) * rho * dx
            solve(lhs(b) == rhs(b), phi0, solver_parameters=direct_parameters)

        return phi0