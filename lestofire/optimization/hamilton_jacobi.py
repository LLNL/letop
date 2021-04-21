from firedrake import (
    MAX,
    READ,
    CellDiameter,
    Constant,
    File,
    Function,
    Max,
    TestFunction,
    TrialFunction,
    derivative,
    diag,
    dot,
    ds,
    dS,
    dx,
    inner,
    par_loop,
    replace,
    solve,
    sqrt,
)
import math
from firedrake.bcs import DirichletBC
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import ExtrudedMeshTopology
from firedrake.ufl_expr import FacetNormal
from firedrake.utility_meshes import UnitSquareMesh
from firedrake.variational_solver import (
    LinearVariationalProblem,
    LinearVariationalSolver,
)
from pyadjoint import no_annotations
from pyadjoint.tape import stop_annotating
from ufl import VectorElement, as_vector
from ufl.algebra import Abs
from ufl import lhs, rhs
from pyop2.profiling import timed_function


def calculate_max_vel(velocity):
    mesh = velocity.ufl_domain()
    MAXSP = FunctionSpace(mesh, "R", 0)
    maxv = Function(MAXSP)
    domain = "{[i, j] : 0 <= i < u.dofs}"
    instruction = f"""
                    maxv[0] = abs(u[i, 0]) + abs(u[i, 1])
                    """
    par_loop(
        (domain, instruction),
        dx,
        {"u": (velocity, READ), "maxv": (maxv, MAX)},
        is_loopy_kernel=True,
    )
    maxval = maxv.dat.data[0]
    return maxval


direct_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

iterative_parameters = {
    "ksp_type": "fgmres",
    # "ksp_monitor_true_residual": None,
    "ksp_max_it": 2000,
    "ksp_atol": 1e-9,
    "ksp_rtol": 1e-9,
    "pc_type": "jacobi",
    "ksp_converged_reason": None,
}


def check_elem_fe(elem_fe):
    supported_fe = ["DQ", "Discontinuous Lagrange"]
    if elem_fe.family() == "TensorProductElement":
        sub_elem = elem_fe.sub_elements()
        if (
            sub_elem[0].family() not in supported_fe
            or sub_elem[1].family() not in supported_fe
        ):
            raise RuntimeError(
                "Only Discontinuous Galerkin function space for extruded elements is supported"
            )
    else:
        if elem_fe.family() not in supported_fe:
            raise RuntimeError(
                "Only Discontinuous Galerkin function space is supported"
            )


class HJLocalDG(object):
    def __init__(self, bcs=None, f=Constant(0.0), hmin=None, n_steps=1):
        """Solver for the Hamilton Jacobi (HJ) PDE with a Local Discontinuous Galerkin method based on
            Jue Yan, Stanley Osher,
            A local discontinuous Galerkin method for directly solving Hamilton–Jacobi equations,
            Journal of Computational Physics,
            Volume 230, Issue 1,
            2011,
            Pages 232-244,

        Args:
            mesh ([type]): Domain mesh
            PHI ([type]): Level set Function space
            bcs ([type], optional): Dirichlet boundary conditions. Defaults to None.
            f ([type], optional): Source term in the HJ PDE. Defaults to Constant(0.0).
            hmin ([type], optional): Minimum mesh element size.
                                    If set, it will calculate the time step size for stability.
                                    Defaults to None.
            n_steps (int, optional): Number of time steps. Defaults to 1.
        """
        self.f = f
        self.bcs = bcs
        self.hmin = hmin
        self.dt = 1.0
        self.n_steps = n_steps

    @timed_function("Solve Hamilton-Jacobi solver")
    @no_annotations
    def solve(self, velocity, phin, scaling=1.0):
        """Solve the HJ PDE for self.n_steps steps

        Args:
            velocity ([type]): Advection direction
            phin ([type]): Initial level set function
            scaling (float, optional): Time step scaling. Defaults to 1.0.

        Returns:
            [type]: Updated level set function
        """
        from functools import partial

        import numpy as np

        phi0 = phin.copy(deepcopy=True)
        DG0 = phi0.function_space()
        check_elem_fe(DG0.ufl_element())
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

        from ufl import diag, div

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

        jacobi_solver = {
            "ksp_type": "preonly",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }

        def H(p):
            return inner(velocity, p)

        def dHdp(p):
            return Abs(velocity)

        def alpha(p1, p2):
            return as_vector(
                [
                    Max(dHdp_p1_i, dHdp_p2_i)
                    for dHdp_p1_i, dHdp_p2_i in zip(dHdp(p1), dHdp(p2))
                ]
            )

        p1 = Function(VDG0)
        p2 = Function(VDG0)

        if self.hmin:
            maxv = calculate_max_vel(velocity)
            self.dt = self.hmin / maxv
            self.n_steps = math.ceil(scaling / self.dt)
        else:
            self.dt = scaling
        dt = self.dt
        print(f"time step: {self.dt}, n_steps: {self.n_steps}")

        problem_phi_1 = LinearVariationalProblem(lhs(a1), rhs(a1), p1)
        solver_phi_1 = LinearVariationalSolver(
            problem_phi_1, solver_parameters=jacobi_solver
        )

        problem_phi_2 = LinearVariationalProblem(lhs(a2), rhs(a2), p2)
        solver_phi_2 = LinearVariationalSolver(
            problem_phi_2, solver_parameters=jacobi_solver
        )

        b = (phi - phi0) * rho / Constant(dt) * dx + (
            H((p1 + p2) / Constant(2.0))
            - Constant(1.0 / 2.0) * inner(alpha(p1, p2), (p1 - p2))
        ) * rho * dx
        problem_phi0 = LinearVariationalProblem(lhs(b), rhs(b), phi0)
        solver_phi0 = LinearVariationalSolver(
            problem_phi0, solver_parameters=jacobi_solver
        )

        for j in range(self.n_steps):
            solver_phi_1.solve()
            solver_phi_2.solve()
            solver_phi0.solve()
        return phi0
