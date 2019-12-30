from firedrake import *
from firedrake.petsc import PETSc
from ufl import conditional, And

distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 3)}

from parameters_heat_exch import (INMOUTH2, INMOUTH1, line_sep, dist_center, inlet_width,
                                WALLS, INLET1, INLET2, OUTLET1, OUTLET2, width, ymax1)

refinements = [2]
nus = [1]
data = []
for nref in refinements:
    mesh = Mesh("./3D_mesh.msh", distribution_parameters=distribution_parameters)
    mh = MeshHierarchy(mesh, nref, distribution_parameters=distribution_parameters)
    mesh = mh[-1]


    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = MixedFunctionSpace([V, Q])

    PETSc.Sys.Print("Using %d refinements (%d dofs)" % (nref, Z.dim()))
    nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

    z_sol = Function(Z)
    (u, p) = split(z_sol)
    (v, q) = split(TestFunction(Z))
    eps = lambda v: sym(grad(v))
    nu = Constant(0)

    (x, y, z) = SpatialCoordinate(mesh)

    from ufl import gt, lt
    alpha = Constant(1.0)*conditional(
                            And(lt(y, 1.2),
                                And(gt(y, -0.4),
                                    And(lt(x, 0.8), gt(x, 0.3)))), 1, 0)
    File("alpha.pvd").write(interpolate(alpha, Q))

    F = (
        + nu*inner(eps(u), eps(v))*dx
        - inner(p, div(v))*dx
        - inner(q, div(u))*dx
        + alpha*inner(u, v)*dx
    )

    u_inflow = 2e-1
    inflow1 = as_vector([u_inflow*sin(z * pi / inlet_width), sin((y - ymax1) * pi / inlet_width), 0.0])
    noslip = Constant((0.0, 0.0, 0.0))
    bcs1_1 = DirichletBC(Z.sub(0), noslip, WALLS)
    bcs1_2 = DirichletBC(Z.sub(0), inflow1, INLET1)
    bcs1_3 = DirichletBC(Z.sub(1), Constant(0.0), OUTLET1)
    bcs1_4 = DirichletBC(Z.sub(0), noslip, INLET2)
    bcs1_5 = DirichletBC(Z.sub(0), noslip, OUTLET2)
    bcs = [bcs1_1,bcs1_2,bcs1_3,bcs1_4, bcs1_5]

    # Configure here
    eigenvalue_estimates = (0.8, 8.5)
    r_weight = 0.1
    inclusive = True
    natural_weights = False
    chebyshev_its = 1
    # End configuration

    parameters = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "ksp_norm_type": "unpreconditioned",
        "ksp_monitor": None,
        "ksp_converged_reason": None,
        "ksp_rtol": 1.0e-10,
        "ksp_max_it": 200,
        "pc_type": "mg",
        "pc_mg_cycle_type": "v",
        "pc_mg_type": "multiplicative",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_ksp_convergence_test": "skip",
        "mg_levels_ksp_max_it": 2,
        "mg_levels_pc_type": "python",
        "mg_levels_pc_python_type": "firedrake.PatchPC",
        "mg_levels_patch_pc_patch_save_operators": True,
        "mg_levels_patch_pc_patch_partition_of_unity": False,
        "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
        "mg_levels_patch_pc_patch_construct_dim": 0,
        "mg_levels_patch_pc_patch_construct_type": "vanka",
        "mg_levels_patch_pc_patch_exclude_subspaces": "1",
        "mg_levels_patch_sub_ksp_type": "preonly",
        "mg_levels_patch_sub_pc_type": "lu",
        "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
        "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    }

    problem = NonlinearVariationalProblem(F, z_sol, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=parameters, options_prefix="", nullspace=nsp)
    linear_its = []
    for nu_ in nus:
        nu.assign(nu_)
        PETSc.Sys.Print("Solving for nu = %s" % float(nu))
        with z_sol.dat.vec_wo as x:
            x.setRandom()
        solver.solve()
        u1, _ = z_sol.split()
        u1.rename("Velocity")
        File("u1.pvd").write(u1)
        linear_its.append(solver.snes.getLinearSolveIterations())
    data.append((nref, Z.dim(), linear_its))


def format_its(its):
    return "{}".format(its)


def format_dof(dof):
    for scale in range(1, 10):
        if dof < 10**scale:
            return "$%1.2f \\times 10^{%d}$" % (dof/(10**(scale-1)), scale-1)


def format_nu(nu):
    import math
    return "$10^{}$".format(round(math.log(nu, 10)))


if COMM_WORLD.rank == 0:
    import json
    basename = "table-stokes"
    with open(f"{basename}.tex", "w") as f:
        f.write("\\begin{tabular}{cc|ccccc}\n")
        f.write("\\toprule\n")
        f.write("\\# refinements & \\# degrees of freedom & \\multicolumn{5}{c}{$\\nu$} \\\\\n")
        f.write(" && {} \\\\\n".format(" & ".join(map(format_nu, nus))))
        f.write("\\midrule\n")
        for ref, dof, its in data:
            f.write(f"{ref} & {format_dof(dof)} & {' & '.join(map(format_its, its))}")
            f.write(r"\\")
            f.write("\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    with open(f"{basename}.json", "w") as f:
        f.write(json.dumps({"header": ("nref", "ndof", "linear_its"),
                            "nus": nus,
                            "data": data}))

