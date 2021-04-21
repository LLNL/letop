from firedrake import (
    LinearVariationalSolver,
    LinearVariationalProblem,
    FacetNormal,
    VectorElement,
    FunctionSpace,
    Function,
    Constant,
    TrialFunction,
    TestFunction,
    assemble,
    solve,
    par_loop,
    DirichletBC,
    WRITE,
    READ,
    RW,
    File,
    warning,
)
from pyadjoint.enlisting import Enlist
from pyadjoint import no_annotations
from firedrake.mesh import ExtrudedMeshTopology
from lestofire.physics import min_mesh_size

from ufl import grad, inner, dot, dx, ds
from ufl import ds_b, ds_t, ds_tb, ds_v


direct_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

iterative_parameters = {
    "mat_type": "aij",
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 200,
    "pc_hypre_boomeramg_coarsen_type": "HMIS",
    "pc_hypre_boomeramg_agg_nl": 1,
    "pc_hypre_boomeramg_strong_threshold": 0.7,
    "pc_hypre_boomeramg_interp_type": "ext+i",
    "pc_hypre_boomeramg_P_max": 4,
    "pc_hypre_boomeramg_relax_type_all": "sequential-Gauss-Seidel",
    "pc_hypre_boomeramg_grid_sweeps_all": 1,
    "pc_hypre_boomeramg_max_levels": 15,
    "ksp_monitor_true_residual": None,
}


class RegularizationSolver(object):
    @no_annotations
    def __init__(
        self,
        S,
        mesh,
        beta=1,
        gamma=1.0e4,
        bcs=None,
        dx=dx,
        design_domain=None,
        solver_parameters=direct_parameters,
        output_dir="./",
    ):
        """
        Solver class to regularize the shape derivatives as explained in
        Frédéric de Gournay
        Velocity Extension for the Level-set Method and Multiple Eigenvalues in Shape Optimization
        SIAM J. Control Optim., 45(1), 343–367. (25 pages)
        Args:
            S ([type]): Function space of the mesh coordinates
            mesh ([type]): Mesh
            beta ([type], optional): Regularization parameter.
                                    It should be finite multiple of the mesh size.
                                    Defaults to 1.
            gamma ([type], optional): Penalty parameter for the penalization of the normal components
                                      of the regularized shape derivatives on the boundary. Defaults to 1.0e4.
            bcs ([type], optional): Dirichlet Boundary conditions.
                                    They should be setting the regularized shape derivatives to zero
                                    wherever there are boundary conditions on the original PDE.
                                    Defaults to None.
            dx ([type], optional): [description]. Defaults to dx.
            design_domain ([type], optional): If we're interested in setting the shape derivatives to
                                              zero outside of the design_domain,
                                              we pass design_domain marker.
                                              This is convenient when we want to fix certain regions in the domain.
                                              Defaults to None.
            solver_parameters ([type], optional): Solver options. Defaults to direct_parameters.
            output_dir (str, optional): Plot the output somewhere. Defaults to "./".
        """
        n = FacetNormal(mesh)
        theta, xi = [TrialFunction(S), TestFunction(S)]
        self.xi = xi

        hmin = min_mesh_size(mesh)
        if beta > 20.0 * hmin:
            warning(
                f"Length scale parameter beta ({beta}) is much larger than the mesh size {hmin}"
            )

        self.beta_param = Constant(beta)

        self.a = (self.beta_param * inner(grad(theta), grad(xi)) + inner(theta, xi)) * (
            dx
        )
        if isinstance(mesh.topology, ExtrudedMeshTopology):
            ds_reg = ds_b + ds_v + ds_tb + ds_t
        else:
            ds_reg = ds
        self.a += Constant(gamma) * (inner(dot(theta, n), dot(xi, n))) * ds_reg

        # Dirichlet boundary conditions equal to zero for regions where we want
        # the domain to be static, i.e. zero velocities

        if bcs is None:
            self.bcs = []
        else:
            self.bcs = Enlist(bcs)
        if design_domain is not None:
            # Heaviside step function in domain of interest
            V_DG0_B = FunctionSpace(mesh, "DG", 0)
            I_B = Function(V_DG0_B)
            I_B.assign(1.0)
            # Set to zero all the cells within sim_domain
            par_loop(
                ("{[i] : 0 <= i < f.dofs}", "f[i, 0] = 0.0"),
                dx(design_domain),
                {"f": (I_B, WRITE)},
                is_loopy_kernel=True,
            )

            I_cg_B = Function(S)
            dim = S.mesh().geometric_dimension()
            # Assume that `A` is a :class:`.Function` in CG1 and `B` is a
            #:class:`.Function` in DG0. Then the following code sets each DoF in
            # `A` to the maximum value that `B` attains in the cells adjacent to
            # that DoF::
            par_loop(
                (
                    "{{[i, j] : 0 <= i < A.dofs and 0 <= j < {0} }}".format(dim),
                    "A[i, j] = fmax(A[i, j], B[0, 0])",
                ),
                dx,
                {"A": (I_cg_B, RW), "B": (I_B, READ)},
                is_loopy_kernel=True,
            )

            import numpy as np

            class MyBC(DirichletBC):
                def __init__(self, V, value, markers):
                    # Call superclass init
                    # We provide a dummy subdomain id.
                    super(MyBC, self).__init__(V, value, 0)
                    # Override the "nodes" property which says where the boundary
                    # condition is to be applied.
                    self.nodes = np.unique(
                        np.where(markers.dat.data_ro_with_halos > 0)[0]
                    )

            self.bcs.append(MyBC(S, 0, I_cg_B))

        self.Av = assemble(self.a, bcs=self.bcs)

        self.solver_parameters = solver_parameters

    @no_annotations
    def solve(self, velocity, dJ):
        """Solve the problem

        Args:
            velocity ([type]): Solution. velocity means regularized shape derivatives
            dJ ([type]): Shape derivatives to be regularized
        """
        for bc in self.bcs:
            bc.apply(dJ)

        solve(
            self.Av,
            velocity.vector(),
            dJ,
            options_prefix="reg_solver",
            solver_parameters=self.solver_parameters,
        )