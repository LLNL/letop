from firedrake import (LinearVariationalSolver, LinearVariationalProblem, FacetNormal,
                        VectorElement, FunctionSpace, Function, Constant, TrialFunction,
                        TestFunction, assemble, solve, par_loop, DirichletBC,
                        WRITE, READ, RW, File)
from firedrake_adjoint import stop_annotating

from ufl import (grad, inner, dot, dx, ds)

parameters = {
    "mat_type" : "aij",
    "ksp_type" : "preonly",
    "pc_type" : "lu",
    "pc_factor_mat_solver_type" : "mumps"
}
class RegularizationSolver(object):

    """Solver to regularize the optimization problem"""

    def __init__(self, S, mesh, beta=1e3, gamma=1.0e4, bcs=[], dx=dx, sim_domain=None):
        n = FacetNormal(mesh)
        theta,xi = [TrialFunction(S), TestFunction( S)]
        self.xi = xi

        self.a = (Constant(beta)*inner(grad(theta),grad(xi)) + inner(theta,xi))*(dx) + \
               Constant(gamma)*(inner(dot(theta,n),dot(xi,n)) * ds)

        # Dirichlet boundary conditions equal to zero for regions where we want
        # the domain to be static, i.e. zero velocities

        self.bcs = bcs
        if sim_domain is not None:
            # Heaviside step function in domain of interest
            V_DG0_B = FunctionSpace(mesh, "DG", 0)
            I_B = Function(V_DG0_B)
            par_loop(('{[i] : 0 <= i < f.dofs}', 'f[i, 0] = 1.0'),
                     dx(sim_domain), {'f': (I_B, WRITE)}, is_loopy_kernel=True)

            I_cg_B = Function(S)
            dim = S.mesh().geometric_dimension()
            par_loop(('{{[i, j] : 0 <= i < A.dofs and 0 <= j < {0} }}'.format(dim), 'A[i, j] = fmax(A[i, j], B[0, 0])'),
                     dx, {'A' : (I_cg_B, RW), 'B': (I_B, READ)}, is_loopy_kernel=True)

            import numpy as np
            class MyBC(DirichletBC):
                def __init__(self, V, value, markers):
                    # Call superclass init
                    # We provide a dummy subdomain id.
                    super(MyBC, self).__init__(V, value, 0)
                    # Override the "nodes" property which says where the boundary
                    # condition is to be applied.
                    self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])

            self.bcs.append(MyBC(S, 0, I_cg_B ))

        self.Av = assemble(self.a, bcs=self.bcs)

        self.beta_pvd = File("beta.pvd")

    def solve(self, velocity, dJ, solver_parameters=parameters):

        with dJ.dat.vec as v:
            v *= -1.0
        with stop_annotating():
            for bc in self.bcs:
                bc.apply(dJ)

        with stop_annotating():
            assemble(self.a, bcs=self.bcs, tensor=self.Av)
            solve(self.Av, velocity.vector(), dJ, solver_parameters=solver_parameters)
        self.beta_pvd.write(velocity)
