import firedrake as fd
from firedrake import inner, sqrt, grad, dx, conditional, le
from pyop2.base import READ, RW
import numpy as np


class BCOut(fd.DirichletBC):
    def __init__(self, V, value, markers):
        super(BCOut, self).__init__(V, value, 0)
        self.nodes = np.unique(
            np.where(markers.dat.data_ro_with_halos == 0)[0]
        )


class BCInt(fd.DirichletBC):
    def __init__(self, V, value, markers):
        super(BCInt, self).__init__(V, value, 0)
        self.nodes = np.unique(
            np.where(markers.dat.data_ro_with_halos != 0)[0]
        )


class ReinitSolverCG:
    def __init__(self, V: fd.FunctionSpace) -> None:
        """Reinitialization solver. Returns the level set
        to a signed distance function

        Args:
            V (fd.FunctionSpace): Function space of the level set
        """
        self.V = V
        self.mesh = V.ufl_domain()
        self.DG0 = fd.FunctionSpace(self.mesh, "DG", 0)
        self.phi = fd.Function(V)

        rho, sigma = fd.TrialFunction(V), fd.TestFunction(V)
        a = rho * sigma * dx
        self.phi_int = fd.Function(V)
        self.A_proj = fd.assemble(a)

    def solve(self, phi: fd.Function, iters: int = 30) -> fd.Function:

        marking = fd.Function(self.DG0)
        marking_bc_nodes = fd.Function(self.V)
        # Mark cells cut by phi(x) = 0
        domain = "{[i, j]: 0 <= i < b.dofs}"
        instructions = """
                <float64> min_value = 1e20
                <float64> max_value = -1e20
                for i
                    min_value = fmin(min_value, b[i, 0])
                    max_value = fmax(max_value, b[i, 0])
                end
                a[0, 0] = 1.0 if (min_value < 0 and max_value > 0) else 0.0
                """
        fd.par_loop(
            (domain, instructions),
            dx,
            {"a": (marking, RW), "b": (phi, READ)},
            is_loopy_kernel=True,
        )
        # Mark the nodes in the marked cells
        fd.par_loop(
            ("{[i] : 0 <= i < A.dofs}", "A[i, 0] = fmax(A[i, 0], B[0, 0])"),
            dx,
            {"A": (marking_bc_nodes, RW), "B": (marking, READ)},
            is_loopy_kernel=True,
        )
        # Mark the nodes in the marked cells
        # Project the gradient of phi on the cut cells
        self.phi.assign(phi)
        V = self.V
        rho, sigma = fd.TrialFunction(V), fd.TestFunction(V)
        a = rho * sigma * marking * dx
        L_proj = (
            self.phi
            / sqrt(inner(grad(self.phi), grad(self.phi)))
            * marking
            * sigma
            * dx
        )
        bc_proj = BCOut(V, fd.Constant(0.0), marking_bc_nodes)
        self.A_proj = fd.assemble(a, tensor=self.A_proj, bcs=bc_proj)
        b_proj = fd.assemble(L_proj, bcs=bc_proj)
        parameters = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_monitor": None,
        }
        solver_proj = fd.LinearSolver(
            self.A_proj, solver_parameters=parameters
        )
        solver_proj.solve(self.phi_int, b_proj)

        def nabla_phi_bar(phi):
            return sqrt(inner(grad(phi), grad(phi)))

        def d1(s):
            return fd.Constant(1.0) - fd.Constant(1.0) / s

        def d2(s):
            return conditional(
                le(s, fd.Constant(1.0)), s - fd.Constant(1.0), d1(s)
            )

        def residual_phi(phi):
            return fd.norm(
                fd.assemble(
                    d2(nabla_phi_bar(phi)) * inner(grad(phi), grad(sigma)) * dx
                )
            )

        a = inner(grad(rho), grad(sigma)) * dx
        L = (
            inner(
                (-d2(nabla_phi_bar(self.phi)) + fd.Constant(1.0))
                * grad(self.phi),
                grad(sigma),
            )
            * dx
        )
        bc = BCInt(V, self.phi_int, marking_bc_nodes)
        phi_sol = fd.Function(V)
        A = fd.assemble(a, bcs=bc)
        b = fd.assemble(L, bcs=bc)
        solver = fd.LinearSolver(A, solver_parameters=parameters)

        # Solve the Signed distance equation with Picard iteration
        bc.apply(phi_sol)
        for _ in range(iters):
            solver.solve(phi_sol, b)
            self.phi.assign(phi_sol)
            res = residual_phi(phi_sol)
            print(f"Residual norm: {res}")
            b = fd.assemble(L, bcs=bc, tensor=b)
        return self.phi
