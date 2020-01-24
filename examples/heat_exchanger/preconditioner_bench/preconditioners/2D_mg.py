from firedrake import Mesh, MeshHierarchy, DistributedMeshOverlapType
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 3)}
nref = 2
mesh = Mesh("./2D_mesh_mg.msh", distribution_parameters=distribution_parameters)
mh = MeshHierarchy(mesh, nref, distribution_parameters=distribution_parameters, reorder=True)
mesh = mh[-1]
stokes_parameters = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "ksp_norm_type": "unpreconditioned",
        #"ksp_monitor": None,
        #"ksp_converged_reason": None,
        "ksp_rtol": 1.0e-10,
        "ksp_max_it": 1000,
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
temperature_parameters = {
        "ksp_type": "fgmres",
        "ksp_max_it": 200,
        "ksp_rtol": 1e-12,
        "ksp_atol": 1e-7,
        "pc_type": "mg",
        "pc_mg_type": "full",
        #"ksp_converged_reason" : None,
        #"ksp_monitor_true_residual" : None,
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "sor",
}

