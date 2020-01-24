from firedrake import Mesh
mesh = Mesh("./2D_mesh.msh")
fieldsplit_0_gamg = {
        "ksp_type" : "preonly",
        "pc_type" : "gamg",
        "pc_gamg_type" : "agg",
        #"ksp_monitor_true_residual": None,
        "mg_levels_esteig_ksp_type" : "cg",
        "mg_levels_ksp_type" : "chebyshev",
        "mg_levels_ksp_chebyshev_esteig_steps" : 10,
        "mg_levels_pc_type" : "sor",
        "pc_gamg_agg_nsmooths" : 2,
        "pc_gamg_threshold" : 0.5, # 0.4 working before
}
stokes_parameters = {
    "mat_type" : "aij",
    #"ksp_converged_reason": None,
    "ksp_max_it" : 2000,
    "ksp_norm_type" : "unpreconditioned",
    "ksp_atol" : 1e-9,
    "ksp_type" : "gmres",
    "pc_type" : "fieldsplit",
    "pc_fieldsplit_type" : "schur",
    #"ksp_monitor_true_residual": None,
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_schur_precondition": "selfp" ,
    "pc_fieldsplit_detect_saddle_point": None,
    "fieldsplit_0": fieldsplit_0_gamg,
    "fieldsplit_1" : fieldsplit_0_gamg
}
temperature_parameters = {
    "ksp_type" : "fgmres",
    "ksp_max_it": 2000,
    "pc_type" : "hypre",
    #"ksp_monitor_true_residual": None,
    "ksp_gmres_restart" : 500,
    "ksp_gmres_modifiedgramschmidt": None,
    "pc_hypre_type" : "boomeramg",
    "pc_hypre_boomeramg_coarsen_type" : "HMIS",
    "ksp_atol" : 1e-10,
    "ksp_rtol" : 1e-10,
    "pc_mg_cycles" : 4,
    #"ksp_converged_reason": None,
    "snes_monitor": None,
    "pc_mg_type" : "full"
}
