# Preconditioning saddle-point systems
# ====================================
#
# Introduction
# ------------
#
# In this demo, we will discuss strategies for solving saddle-point
# systems using the mixed formulation of the Poisson equation introduced
# :doc:`previously </demos/poisson_mixed.py>` as a concrete example.
# Such systems are somewhat tricky to precondition effectively, modern
# approaches typically use block-factorisations.  We will encounter a
# number of methods in this tutorial.  For many details and background
# on solution methods for saddle point systems, :cite:`Benzi:2005` is a
# nice review.  :cite:`Elman:2014` is an excellent text with a strong
# focus on applications in fluid dynamics.
#
# We start by repeating the formulation of the problem.  Starting from
# the primal form of the Poisson equation, :math:`\nabla^2 u = -f`, we
# introduce a vector-valued flux, :math:`\sigma = \nabla u`.  The
# problem then becomes to find :math:`u` and :math:`\sigma` in some
# domain :math:`\Omega` satisfying
#
# .. math::
#
#    \sigma - \nabla u &= 0 \quad &\textrm{on}\ \Omega\\
#    \nabla \cdot \sigma &= -f \quad &\textrm{on}\ \Omega\\
#    u &= u_0 \quad &\textrm{on}\ \Gamma_D\\
#    \sigma \cdot n &= g \quad &\textrm{on}\ \Gamma_N
#
# for some specified function :math:`f`.  We now seek :math:`(u, \sigma)
# \in V \times \Sigma` such that
#
# .. math::
#
#    \int_\Omega \sigma \cdot \tau + (\nabla \cdot \tau)\, u\,\mathrm{d}x
#    &= \int_\Gamma (\tau \cdot n)\,u\,\mathrm{d}s &\quad \forall\ \tau
#    \in \Sigma, \\
#    \int_\Omega (\nabla \cdot \sigma)\,v\,\mathrm{d}x
#    &= -\int_\Omega f\,v\,\mathrm{d}x &\quad \forall\ v \in V.
#
# A stable choice of discrete spaces for this problem is to pick
# :math:`\Sigma_h \subset \Sigma` to be the lowest order Raviart-Thomas
# space, and :math:`V_h \subset V` to be the piecewise constants,
# although this is :doc:`not the only choice </demos/poisson_mixed.py>`.
# For ease of exposition we choose the domain to be the unit square, and
# enforce homogeneous Dirichlet conditions on all walls.  The forcing
# term is chosen to be random.
#
# Globally coupled elliptic problems, such as the Poisson problem,
# require effective preconditioning to attain *mesh independent*
# convergence.  By this we mean that the number of iterations of the
# linear solver does not grow when the mesh is refined.  In this demo,
# we will study various ways to achieve this in Firedrake.
#
# As ever, we begin by importing the Firedrake module::

from firedrake import *

# Bulding the problem
# -------------------
#
# Rather than defining a mesh and function spaces straight away, since
# we wish to consider the effect that mesh refinement has on the
# performance of the solver, we instead define a Python function which
# builds the problem we wish to solve.  This takes as arguments the size
# of the mesh, the solver parameters we wish to apply, an optional
# parameter specifying a "preconditioning" operator to apply, and a
# final optional argument specifying whether the block system should be
# assembled as a single "monolithic" matrix or a :math:`2 \times 2`
# block of smaller matrices. ::

def build_problem(mesh_size, parameters, block_matrix=False):
    distribution = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    N = 5
    base = BoxMesh(N, N, N, 2, 2, 2, distribution_parameters=parameters)
    mh = MeshHierarchy(base, 1, distribution_parameters=distribution)
    mesh = mh[-1]

    Sigma = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    W = Sigma * V

# Having built the function spaces, we can now proceed to defining the
# problem.  We will need some trial and test functions for the spaces::

#
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

# along with a function to hold the forcing term, living in the
# discontinuous space. ::

#
    f = Function(V)

# To initialise this function to a random value we access its :class:`~.Vector`
# form and use numpy_ to set the values::

#
    import numpy as np
    fvector = f.vector()
    fvector.set_local(np.random.uniform(size=fvector.local_size()))

# Note that the homogeneous Dirichlet conditions in the primal
# formulation turn into homogeneous Neumann conditions on the dual
# variable and we therefore drop the surface integral terms in the
# variational formulation (they are identically zero).  As a result, the
# specification of the variational problem is particularly simple::

#
    a = dot(sigma, tau)*dx + div(tau)*u*dx + div(sigma)*v*dx
    L = -f*v*dx

    if block_matrix:
        mat_type = 'nest'
    else:
        mat_type = 'aij'

    def riesz(W):
        sigma, u = TrialFunctions(W)
        tau, v = TestFunctions(W)

        return (dot(sigma, tau) + div(sigma)*div(tau) + u*v)*dx

    u_sol = Function(W)
    #problem = LinearVariationalProblem(a, L, u_sol, aP=riesz(W))
    #solver = LinearVariationalSolver(problem, solver_parameters=parameters)
    aP = riesz(W)


    return a, L, aP, u_sol


# Now we set up the solver parameters.  We will still use a
# ``fieldsplit`` preconditioner, but this time it will be additive,
# rather than a Schur complement. ::

parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "ksp_max_it": 100,
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",

# Now we choose how to invert the two blocks.  The second block is easy,
# it is just a mass matrix in a discontinuous space and is therefore
# inverted exactly using a single application of zero-fill ILU. ::

#
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "ilu",

# The :math:`H(\text{div})` inner product is the tricky part.  In fact,
# we currently do not have a good way of inverting this in Firedrake.
# For now we will invert it with a direct solver.  This is a reasonable
# option up to a few tens of thousands of degrees of freedom. ::

#
    "fieldsplit_0_" : {
                "ksp_type": "preonly",
                "ksp_converged_reason": None,
                #"pc_type": "lu",
                "pc_type": "mg",
                #"pc_mg_type": "full",
                "mg_levels_ksp_type": "richardson",
                "mg_levels_ksp_norm_type": "unpreconditioned",
                "mg_levels_ksp_richardson_scale": 1.0/3.0,
                #"mg_levels_ksp_max_it": 1,
                "mg_levels_ksp_convergence_test": "skip",
                "mg_levels_ksp_convergence_reason": None,
                "mg_levels_pc_type": "python",
                "mg_levels_pc_python_type": "firedrake.PatchPC",
                #"mg_levels_patch_pc_patch_save_operators": True,
                #"mg_levels_patch_pc_patch_partition_of_unity": False,
                "mg_levels_patch_pc_patch_construct_type": "star",
                "mg_levels_patch_pc_patch_construct_dim": 0,
                "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
                "mg_levels_patch_sub_ksp_type": "preonly",
                "mg_levels_patch_sub_pc_type": "lu",
                "mg_coarse_pc_type": "python",
                "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                "mg_coarse_assembled_pc_type": "lu",
                "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
               }
}

# .. note::
#
#    For larger problems, you will probably need to use a sparse direct
#    solver such as MUMPS_, which may be selected by additionally
#    specifying ``"fieldsplit_0_pc_factor_mat_solver_type": "mumps"``.
#
#    To use MUMPS_ you will need to have configured PETSc_ appropriately
#    (using at the very least ``--download-mumps``).
#
# Let's see what the iteration count looks like now. ::

print("Riesz-map preconditioner")
for n in range(8):
    a, L, aP, u_sol = build_problem(n, parameters, block_matrix=True)
    solve(a==L, u_sol, Jp=aP, solver_parameters=parameters)
    print(solver.snes.getIterationNumber())

# ============== ==================
#    2                  3
#    8                  5
#    32                 5
#    128                5
#    512                5
#    2048               5
#    8192               5
#    32768              5
# ============== ==================
#
# Providing access to scalable preconditioners for these kinds of
# problems is currently a wishlist item for Firedrake.  There are two
# options, either geometric multigrid with strong,
# Schwarz-based, smoothers :cite:`Arnold:2000`.  Or else algebraic
# multigrid approaches using the auxiliary-space preconditioning method
# of :cite:`Hiptmair:2007`.  Support for the algebraic approach is
# available in hypre_ (the AMS and AMR preconditioners), and an
# interface exists in PETSc_.  If you're interested in adding the
# missing pieces to support this in Firedrake, we would :doc:`love to
# hear from you </contact>`.
#
# A runnable python script version of this demo is available `here
# <saddle_point_systems.py>`__.
#
# .. rubric:: References
#
# .. bibliography:: demo_references.bib
#    :filter: docname in docnames
#
# .. _PETSc: http://www.mcs.anl.gov/petsc/
# .. _hypre: http://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/software
# .. _PyOP2: http://github.com/OP2/PyOP2/
# .. _numpy: http://www.numpy.org
# .. _MUMPS: http://mumps.enseeiht.fr

