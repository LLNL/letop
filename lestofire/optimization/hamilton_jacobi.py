from firedrake import (
    FunctionSpace,
    TrialFunction,
    TestFunction,
    Constant,
    Function,
    dx,
    ds,
    dS,
    CellDiameter,
    sqrt,
    Max,
    solve,
    dot,
    inner,
    File,
    diag,
    replace,
    derivative,
    par_loop,
    READ,
    MAX,
)
from ufl import as_vector
from firedrake.bcs import DirichletBC
from firedrake.functionspace import VectorFunctionSpace
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


def check_elem_fe(elem_fe):
    supported_fe = ["DQ", "Discontinuous Lagrange"]
    if elem_fe.family() == "TensorProductElement":
        sub_elem = elem_fe.sub_elements()
        if (
            sub_elem[0].family() not in supported_fe
            or sub_elem[0].degree() != 0
            or sub_elem[1].family() not in supported_fe
            or sub_elem[1].degree() != 0
        ):
            raise RuntimeError(
                "Only zero degree Discontinuous Galerkin function space for extruded elements is supported"
            )
    else:
        if elem_fe.family() not in supported_fe or elem_fe.degree() != 0:
            raise RuntimeError(
                "Only zero degree Discontinuous Galerkin function space is supported"
            )


class HJDG(object):
    def __init__(self, mesh, PHI, phi_x0, bcs=None, f=Constant(0.0), hmin=1.0):
        check_elem_fe(PHI.ufl_element())

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

        v = TestFunction(self.PHI)
        ut = TrialFunction(self.PHI)
        u = Function(self.PHI)
        mesh = self.PHI.ufl_domain()
        n = FacetNormal(mesh)

        VDG0 = VectorFunctionSpace(mesh, "DG", 0)
        import numpy as np
        from firedrake import SpatialCoordinate, interpolate

        mesh_coords = np.reshape(
            interpolate(SpatialCoordinate(mesh), VDG0).dat.data_ro,
            (-1, mesh.geometric_dimension()),
        )
        indices_i = np.where(mesh_coords < 0.01)[0]
        m = np.zeros_like(indices_i, dtype=bool)
        m[np.unique(indices_i, return_index=True)[1]] = True
        dof_bottom_corner = indices_i[~m][0]

        beta_interp = interpolate(beta, VDG0)
        from firedrake import assemble

        print(f"Normal components in DG: {assemble(inner(beta_interp, n)*ds)}")
        print(
            f"DG components at bottom corner: {beta_interp.dat.data_ro[dof_bottom_corner]}"
        )

        def F_c(U):
            if hasattr(U, "side"):
                return beta(U.side()) * U
            else:
                return beta * U

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

        from firedrake import dS

        convective_flux.setup(F_c, un("+"), un("-"), n("+"))
        residual_dg = (
            inner(
                convective_flux.interior(F_c, un("+"), un("-"), n("+")),
                (v("+") - v("-")),
            )
            * dS
        )

        maxval = calculate_max_vel(beta)
        self.dt = 1.0 * self.hmin / maxval * scaling
        print(f"maxv: {maxval}, dt: {self.dt}")
        dtc = Constant(self.dt)
        a = ut * v * dx
        L_vel = dtc * (a_term - Constant(0.0) * v * dx)
        L = un * v * dx - L_vel
        u_sol = Function(self.PHI)

        for j in range(steps):
            solve(a == L, u_sol, solver_parameters=direct_parameters)
            print(
                f"bottom corner contribution: {assemble(L_vel).dat.data_ro[dof_bottom_corner] / 1e-4}"
            )
            print(
                f"bottom corner contribution from the interior facets: {assemble(residual_dg).dat.data_ro[dof_bottom_corner] *self.dt / 1e-4}"
            )
            un.assign(u_sol)

            self.phi_viz.assign(un)
            self.phi_pvd.write(self.phi_viz)
            print(f"bottom corner evolution: {un.dat.data_ro[dof_bottom_corner]}")

        return un


class HJLocalDG(object):
    def __init__(self, mesh, PHI, phi_x0, bcs=None, f=Constant(0.0), hmin=1.0):
        check_elem_fe(PHI.ufl_element())
        self.PHI = PHI
        self.mesh = mesh
        self.f = f
        self.phi_x0 = Function(PHI)
        self.bcs = bcs  # TODO Enforce DGDirichletBC type
        from firedrake import H1

        self.phi_pvd = File("./hjdg.pvd", target_continuity=H1)
        self.phi_viz = Function(self.PHI)
        self.dt = 1.0
        self.hmin = hmin

    @no_annotations
    def solve(self, velocity, phin, steps=1, t=0, scaling=1.0):
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

        phi0 = phin.copy(deepcopy=True)
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

        direct_parameters = {
            "snes_type": "ksponly",
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        jacobi_solver = {"ksp_type": "preonly", "pc_type": "jacobi"}

        p1_0 = Function(VDG0)
        p2_0 = Function(VDG0)
        from ufl import lhs, rhs

        solve(lhs(a1) == rhs(a1), p1_0, solver_parameters=jacobi_solver)
        solve(lhs(a2) == rhs(a2), p2_0, solver_parameters=jacobi_solver)

        def H(p):
            return inner(velocity, p)

        def dHdp(p_x, p_y):
            return as_vector([abs(velocity[0]), abs(velocity[1])])

        def alpha(p1_x, p2_x, p1_y, p2_y):
            return Max(dHdp(p1_x, p1_y)[0], dHdp(p2_x, p2_y)[0])

        def beta(p1_x, p2_x, p1_y, p2_y):
            return Max(dHdp(p1_x, p1_y)[1], dHdp(p2_x, p2_y)[1])

        p1 = Function(VDG0)
        p2 = Function(VDG0)

        maxval = calculate_max_vel(velocity)
        # dt = 1.0 * self.hmin / maxval * scaling
        dt = scaling
        print(f"maxv: {maxval}, dt: {dt}")
        for j in range(steps):

            solve(lhs(a1) == rhs(a1), p1, solver_parameters=jacobi_solver)
            solve(lhs(a2) == rhs(a2), p2, solver_parameters=jacobi_solver)

            if j % 1 == 0:
                if self.phi_pvd:
                    self.phi_pvd.write(phi0)

            b = (phi - phi0) * rho / Constant(dt) * dx + (
                H((p1 + p2) / Constant(2.0))
                - Constant(1.0 / 2.0)
                * alpha(p1[0], p2[0], p1[1], p2[1])
                * (p1[0] - p2[0])
                - Constant(1.0 / 2.0)
                * beta(p1[0], p2[0], p1[1], p2[1])
                * (p1[1] - p2[1])
            ) * rho * dx
            solve(lhs(b) == rhs(b), phi0, solver_parameters=direct_parameters)

        return phi0