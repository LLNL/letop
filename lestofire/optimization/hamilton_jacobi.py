from firedrake import (
    FunctionSpace,
    TrialFunction,
    TestFunction,
    Constant,
    Function,
    FacetArea,
    CellVolume,
    FacetNormal,
    CellDiameter,
    grad,
    jump,
    avg,
    inner,
    dx,
    dS,
    dS_h,
    dS_v,
    solve,
    lhs,
    rhs,
    dot,
    ds,
    div,
    LinearVariationalProblem,
    LinearVariationalSolver,
    VectorFunctionSpace,
    File,
    conditional,
    replace,
    derivative,
    VertexBasedLimiter,
    action,
)
from firedrake.mesh import ExtrudedMeshTopology

from dolfin_dg import LocalLaxFriedrichs, HyperbolicOperator, DGDirichletBC


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


class HJStabSolver(object):
    def __init__(
        self, mesh, PHI, c2_param=0.05, f=Constant(0.0), bc=None, iterative=False
    ):
        self.PHI = PHI
        self.PHI_V = VectorFunctionSpace(mesh, "CG", 1)
        self.mesh = mesh
        self.c2 = c2_param
        self.f = f

        if isinstance(mesh.topology, ExtrudedMeshTopology):
            dS_reg = dS_h + dS_v
            h = CellVolume(mesh) / FacetArea(mesh)
        else:
            dS_reg = dS
            h = CellDiameter(mesh)

        phi = TrialFunction(PHI)
        psi = TestFunction(PHI)
        n = FacetNormal(mesh)

        self.phi_n = Function(PHI)
        self.beta = Function(self.PHI_V)
        self.dt = Constant(1.0)

        self.a = (
            phi / self.dt * psi * dx
            + inner(self.beta, 1.0 / 2.0 * grad(phi)) * psi * dx
            + self.c2
            * avg(h)
            * avg(h)
            * inner(jump(1.0 / 2.0 * grad(phi), n), jump(grad(psi), n))
            * dS_reg
        )

        self.L = (
            self.phi_n / self.dt * psi * dx
            - inner(self.beta, 1.0 / 2.0 * grad(self.phi_n)) * psi * dx
            - self.c2
            * avg(h)
            * avg(h)
            * inner(jump(1.0 / 2.0 * grad(self.phi_n), n), jump(grad(psi), n))
            * dS_reg
            + self.f * psi * dx
        )

        self.phi_sol = Function(self.PHI)
        self.problem = LinearVariationalProblem(self.a, self.L, self.phi_sol, bcs=bc)

        if iterative:
            self.parameters = iterative_parameters
        else:
            self.parameters = direct_parameters

    def solve(self, beta, phi_n, steps=5, t=0, dt=1.0):

        # self.beta.assign(beta, annotate=False)
        self.beta.interpolate(beta)
        self.phi_n.assign(phi_n, annotate=False)
        self.dt.assign(Constant(dt), annotate=False)

        self.solver = LinearVariationalSolver(
            self.problem, solver_parameters=self.parameters, options_prefix="hjsolver_"
        )

        for i in range(steps):
            self.solver.solve(annotate=False)
            self.phi_n.assign(self.phi_sol, annotate=False)

        return self.phi_n


class HJSUPG(object):
    def __init__(self, mesh, PHI, f=Constant(0.0), bc=None):
        self.PHI = PHI
        self.mesh = mesh
        self.f = f
        self.bc = bc

    def solve(self, beta, phi_n, steps=5, t=0, dt=1.0):
        phi = TrialFunction(self.PHI)
        psi = TestFunction(self.PHI)
        n = FacetNormal(self.mesh)
        h = CellDiameter(self.mesh)
        self.f.t = t

        for i in range(steps):
            F = (
                (phi - phi_n) / Constant(dt) * psi * dx
                + inner(beta, 1.0 / 2.0 * grad(phi + phi_n)) * psi * dx
                - self.f * psi * dx
            )

            # Residual
            r = (
                (1 / Constant(dt)) * (phi - phi_n)
                + 1.0 / 2.0 * inner(beta, grad(phi))
                + (1 - 1.0 / 2.0) * inner(beta, grad(phi_n))
                - self.f
            ) * inner(beta, grad(psi))

            # Add SUPG stabilisation terms
            from ufl import sqrt

            vnorm = sqrt(dot(beta, beta))
            tau = 0.5 * h / (2.0 * vnorm)
            tau = 2.0 * 1.0 / (sqrt((1.0 / Constant(dt * dt) + vnorm / h)))
            F += tau * r * dx

            phi_new = Function(self.PHI)

            if self.bc is None:
                solve(lhs(F) == rhs(F), phi_new, annotate=False)
            else:
                solve(lhs(F) == rhs(F), phi_new, bcs=self.bc, annotate=False)
            phi_n.assign(phi_new, annotate=False)

        return phi_n


class HJDG(object):
    def __init__(self, mesh, PHI, phi_x0, f=Constant(0.0)):
        self.PHI = PHI
        self.mesh = mesh
        self.f = f
        self.phi_x0 = Function(PHI)
        self.phi_x0.interpolate(phi_x0)
        self.u_pvd = File("./hjdg.pvd")
        self.u_viz = Function(self.PHI)

    def solve(self, beta, un, steps=1, t=0, dt=1.0, bc=None):

        # Convective Operator
        def F_c(U):
            return beta * U

        v = TestFunction(self.PHI)
        ut = TrialFunction(self.PHI)
        u = Function(self.PHI)
        theta = Constant(0.5)
        uth = theta * u + (1 - theta) * un

        convective_flux = LocalLaxFriedrichs(lambda u, n: dot(beta, n))
        ho = HyperbolicOperator(
            self.mesh, self.PHI, DGDirichletBC(ds, self.phi_x0), F_c, convective_flux
        )
        residual = ho.generate_fem_formulation(u, v)

        a_term = replace(residual, {u: un})
        dtc = Constant(dt)
        F = (u - un) * v * dx + dtc * (a_term - Constant(0.0) * v * dx)
        a = derivative(F, u, ut)
        L = -F

        limiter = VertexBasedLimiter(self.PHI)
        limiter.apply(un)
        for j in range(steps):
            solve(a == L, un, solver_parameters=direct_parameters)
            limiter.apply(un)

            self.u_viz.assign(un)
            self.u_pvd.write(self.u_viz)

        ## Begin the iteration over time steps
        # for j in range(steps):
        #    u1.assign(u)
        #    limiter.apply(u1)
        #    solver.solve()
        #    u1.assign(du1)
        #    limiter.apply(u1)

        #    solver.solve()
        #    u1.assign(0.75 * u + 0.25 * du1)
        #    limiter.apply(u1)
        #    solver.solve()
        #    u.assign((1.0 / 3.0) * u + (2.0 / 3.0) * du1)
        #    limiter.apply(u1)

        #    self.u_viz.assign(u)
        #    self.u_pvd.write(self.u_viz)

        return un
