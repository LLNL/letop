from firedrake import (
    FunctionSpace,
    TrialFunction,
    TestFunction,
    Constant,
    Function,
    dx,
    ds,
    solve,
    dot,
    inner,
    File,
    replace,
    derivative,
    par_loop,
    READ,
    MAX,
)
from firedrake.mesh import ExtrudedMeshTopology
from firedrake.ufl_expr import FacetNormal
from firedrake.utility_meshes import UnitSquareMesh
from pyadjoint import no_annotations

from dolfin_dg import LocalLaxFriedrichs, HyperbolicOperator, DGDirichletBC
from pyadjoint.tape import stop_annotating


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
    # "pc_factor_mat_solver_type": "mumps",
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


class HJDG(object):
    def __init__(self, mesh, PHI, phi_x0, bcs=None, f=Constant(0.0), hmin=1.0):
        self.PHI = PHI
        self.mesh = mesh
        self.f = f
        self.phi_x0 = Function(PHI)
        self.bcs = bcs  # TODO Enforce DGDirichletBC type
        with stop_annotating():
            self.phi_x0.interpolate(phi_x0)
        from firedrake import H1

        self.phi_pvd = File("./hjdg.pvd", target_continuity=H1)
        self.phi_viz = Function(self.PHI)
        self.dt = 1.0
        self.hmin = hmin

    @no_annotations
    def solve(self, beta, un, steps=1, t=0, scaling=1.0):
        # Convective Operator
        def F_c(U):
            if hasattr(U, "side"):
                return beta(U.side()) * U
            else:
                return beta * U

        v = TestFunction(self.PHI)
        ut = TrialFunction(self.PHI)
        u = Function(self.PHI)
        n = FacetNormal(self.PHI.ufl_domain())

        def flux_jacobian(u, n):
            if hasattr(u, "side"):
                return dot(beta(u.side()), n)
            else:
                return dot(beta, n)

        convective_flux = LocalLaxFriedrichs(flux_jacobian_eigenvalues=flux_jacobian)
        ho = HyperbolicOperator(
            self.mesh, self.PHI, self.bcs, F_c=F_c, H=convective_flux
        )
        residual = ho.generate_fem_formulation(u, v)
        residual += inner(dot(F_c(u), n), v) * ds
        a_term = replace(residual, {u: un})

        maxval = calculate_max_vel(beta)
        self.dt = 0.1 * 1.0 * self.hmin / maxval * scaling
        dtc = Constant(self.dt)
        F = (u - un) * v * dx + dtc * (a_term - Constant(0.0) * v * dx)
        a = derivative(F, u, ut)
        L = -F

        for j in range(steps):
            solve(a == L, un, solver_parameters=direct_parameters)

            self.phi_viz.assign(un)
            self.phi_pvd.write(self.phi_viz)

        return un
