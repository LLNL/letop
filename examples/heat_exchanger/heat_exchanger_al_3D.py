from firedrake import *
from firedrake_adjoint import *
import numpy as np
from ufl import replace, conditional
from ufl import min_value, max_value

from lestofire import LevelSetLagrangian, AugmentedLagrangianOptimization, RegularizationSolver

from params3D import (INMOUTH2, INMOUTH1, line_sep, dist_center, inlet_width, ymin1, ymin2,
                                WALLS, INLET1, INLET2, OUTLET1, OUTLET2, width)

def main():
    output_dir = "3D/"

    mesh = Mesh('./3D_mesh.msh')

    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)


    X = SpatialCoordinate(mesh)
    x, y, z = X[0], X[1], X[2]

    phi_expr = -(y - line_sep - 0.05)

    PHI = FunctionSpace(mesh, 'CG', 1)
    phi = interpolate(phi_expr , PHI)
    phi.rename("LevelSet")
    phi_pvd = File(output_dir + "phi_evolution.pvd")
    phi_pvd.write(phi)

    mu = Constant(1e-2)                   # viscosity
    alphamin = 1e-12
    epsilon = Constant(10000.0)
    u_inflow = 2e-1
    tin1 = Constant(10.0)
    tin2 = Constant(100.0)

    iterative = False
    if iterative:
        alphamax = 2.5 * mu / (2e-7)

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
                #"ksp_monitor_true_residual": None,
                "ksp_converged_reason": None,
                "ksp_max_it" : 1000,
                "ksp_norm_type" : "unpreconditioned",
                "ksp_atol" : 1e-9,
                "ksp_type" : "fgmres",
                "pc_type" : "fieldsplit",
                "pc_fieldsplit_type" : "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                "pc_fieldsplit_schur_precondition": "selfp" ,
                "pc_fieldsplit_detect_saddle_point": None,
                "fieldsplit_0": fieldsplit_0_gamg,
                "fieldsplit_1" : fieldsplit_0_gamg
        }
        #stokes_parameters = {
        #    "mat_type" : "aij",
        #    "ksp_type" : "preonly",
        #    "ksp_monitor_true_residual": None,
        #    #"ksp_converged_reason" : None,
        #    "pc_type" : "lu",
        #    "pc_factor_mat_solver_type" : "mumps"
        #}
        # Penalty term
       # temperature_parameters = {
       #             "mat_type" : "aij",
       #             "ksp_monitor_true_residual": None,
       #             "ksp_converged_reason": None,
       #             "ksp_max_it" : 1000,
       #             "ksp_norm_type" : "unpreconditioned",
       #             "ksp_atol" : 1e-9,
       #             "ksp_atol" : 1e-9,
       #             "ksp_type" : "fgmres",
       #             "ksp_gmres_restart" : 1000,
       #             "pc_type" : "gamg",
       #             "pc_mg_type" : "full",
       #             "pc_gamg_type" : "agg",
       #             "pc_gamg_square_graph" : 1,
       #             "ksp_monitor_true_residual": None,
       #             "mg_levels_esteig_ksp_type" : "gmres",
       #             "mg_levels_ksp_type" : "chebyshev",
       #             "mg_levels_ksp_chebyshev_esteig_steps" : 20,
       #             "mg_levels_pc_type" : "sor",
       #             "pc_gamg_agg_nsmooths" : 10,
       #             "pc_gamg_threshold" : 0.1, # 0.4 working before
       #         }
        #temperature_parameters = {
        #        "ksp_type" : "fgmres",
        #        "ksp_max_it": 100000,
        #        "pc_type" : "ml",
        #        "ksp_atol" : 1e-10,
        #        "ksp_norm_type" : "unpreconditioned",
        #        "ksp_rtol" : 1e-10,
        #        "pc_mg_cycles" : 2,
        #        "ksp_monitor_true_residual": None,
        #        "ksp_converged_reason": None,
        #        "snes_monitor": None,
        #        #"pc_mg_type" : "full",
        #        "pc_ml_Threshold" : 0.8,
        #        #"pc_ml_maxNlevels" : 5,
        #        "pc_ml_maxCoarseSize" : 10,
        #    }
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
                "ksp_converged_reason": None,
                "snes_monitor": None,
                "pc_mg_type" : "full"
            }
    else:
        alphamax = 2.5 * mu / (2e-3)
        parameters = {
            "mat_type" : "aij",
            "ksp_type" : "preonly",
            "ksp_monitor_true_residual": None,
            #"ksp_converged_reason" : None,
            "pc_type" : "lu",
            "pc_factor_mat_solver_type" : "mumps"
            }
        stokes_parameters = parameters
        temperature_parameters = parameters


    P2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2*P1
    W = FunctionSpace(mesh, TH)
    print("DOFs: {}".format(W.dim()))

    U = TrialFunction(W)
    u, p = split(U)
    V = TestFunction(W)
    v, q = split(V)

    def hs(phi, epsilon):
        return Constant(alphamax)*Constant(1.0) / ( Constant(1.0) + exp(-epsilon*phi)) + Constant(alphamin)

    def stokes(phi, BLOCK_MOUTH):
        a_fluid = (mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u))
        darcy_term = inner(u, v)
        return a_fluid*dx + hs(phi, epsilon)*darcy_term*dx(0) + alphamax*darcy_term*dx(BLOCK_MOUTH)

    # Dirichelt boundary conditions
    inflow1 = as_vector([u_inflow*sin(z * pi / inlet_width) * sin((y - ymin1) * pi / inlet_width), 0.0, 0.0])
    inflow2 = as_vector([u_inflow*sin(z * pi / inlet_width) * sin((y - ymin2) * pi / inlet_width), 0.0, 0.0])

    noslip = Constant((0.0, 0.0, 0.0))

    # Stokes 1
    bcs1_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs1_2 = DirichletBC(W.sub(0), inflow1, INLET1)
    bcs1_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET1)
    bcs1_4 = DirichletBC(W.sub(0), noslip, INLET2)
    bcs1_5 = DirichletBC(W.sub(0), noslip, OUTLET2)
    bcs1 = [bcs1_1,bcs1_2,bcs1_3,bcs1_4, bcs1_5]

    # Stokes 2
    bcs2_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs2_2 = DirichletBC(W.sub(0), inflow2, INLET2)
    bcs2_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET2)
    bcs2_4 = DirichletBC(W.sub(0), noslip, INLET1)
    bcs2_5 = DirichletBC(W.sub(0), noslip, OUTLET1)
    bcs2 = [bcs2_1,bcs2_2,bcs2_3,bcs2_4, bcs2_5]

    # Forward problems
    U1, U2 = Function(W), Function(W)
    L = inner(Constant((0.0, 0.0, 0.0, 0.0)), V)*dx
    problem = LinearVariationalProblem(stokes(-phi, INMOUTH2), L, U1, bcs=bcs1)
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
    solver_stokes1 = LinearVariationalSolver(problem, solver_parameters=stokes_parameters) #, nullspace=nullspace)
    solver_stokes1.solve()

    u1, _ = Control(U1).tape_value().split()
    u1.rename("Velocity")
    File(output_dir + "u1.pvd").write(u1)

    problem = LinearVariationalProblem(stokes(phi, INMOUTH1), L, U2, bcs=bcs2)
    solver_stokes2 = LinearVariationalSolver(problem, solver_parameters=stokes_parameters)
    solver_stokes2.solve()
    u2, _ = Control(U2).tape_value().split()
    u2.rename("Velocity")
    File(output_dir + "u2.pvd").write(u2)

    # Convection difussion equation
    ks = Constant(1e0)
    cp_value = 5.0e3
    cp = Constant(cp_value)
    T = FunctionSpace(mesh, 'DG', 1)
    t = Function(T, name="Temperature")
    w = TestFunction(T)
    print("DOFs temperature: {}".format(T.dim()))

    # Mesh-related functions
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    u1, p1 = split(U1)
    u2, p2 = split(U2)

    def upwind(u):
        return (dot(u, n) + abs(dot(u, n)))/2.0
    u1n = upwind(u1)
    u2n = upwind(u2)

    # Penalty term
    alpha = Constant(50000.0) # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
    alpha = Constant(50000.0) # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
    # Bilinear form
    a_int = dot(grad(w), ks*grad(t) - cp*(u1 + u2)*t)*dx

    a_fac = Constant(-1.0)*ks*dot(avg(grad(w)), jump(t, n))*dS \
        + Constant(-1.0)*ks*dot(jump(w, n), avg(grad(t)))*dS \
          + ks('+')*(alpha('+')/avg(h))*dot(jump(w, n), jump(t, n))*dS

    a_vel = dot(jump(w), cp*(u1n('+') + u2n('+'))*t('+') - \
            cp*(u1n('-') + u2n('-'))*t('-'))*dS + \
            dot(w, cp*(u1n + u2n)*t)*ds

    a_bnd = dot(w, cp*dot(u1 + u2, n)*t)*(ds(INLET1) + ds(INLET2)) \
            + w*t*(ds(INLET1) + ds(INLET2)) \
            - w*tin1*ds(INLET1) \
            - w*tin2*ds(INLET2) \
            + alpha/h * ks *w*t*(ds(INLET1) + ds(INLET2)) \
            - ks * dot(grad(w), t*n)*(ds(INLET1) + ds(INLET2)) \
            - ks * dot(grad(t), w*n)*(ds(INLET1) + ds(INLET2))

    aT = a_int + a_fac + a_vel + a_bnd

    LT_bnd = alpha/h * ks * tin1 * w * ds(INLET1)  \
            + alpha/h * ks * tin2 * w * ds(INLET2) \
            - tin1 * ks * dot(grad(w), n) * ds(INLET1) \
            - tin2 * ks * dot(grad(w), n) * ds(INLET2)
    eT = aT - LT_bnd

    problem = NonlinearVariationalProblem(eT, t)
    solver_temp = NonlinearVariationalSolver(problem, solver_parameters=temperature_parameters)
    solver_temp.solve()
    File(output_dir + "t.pvd").write(t)


    Power1 = assemble(p1*ds(INLET1)) - 0.1
    Power2 = assemble(p2*ds(INLET2)) - 0.1
    Jform = assemble(Constant(-1e5)*inner(t*u1, n)*ds(OUTLET1))

    print("Power drop 1 {:.5f}".format(Power1))
    print("Power drop 2 {:.5f}".format(Power2))

    U1control = Control(U1)
    U2control = Control(U2)
    u1_pvd = File(output_dir + "u1.pvd")
    u2_pvd = File(output_dir + "u2.pvd")
    def deriv_cb(phi):
        phi_pvd.write(phi[0])
        u1, _ = U1control.tape_value().split()
        u2, _ = U2control.tape_value().split()
        u1.rename("Velocity")
        u2.rename("Velocity")
        u1_pvd.write(u1)
        u2_pvd.write(u2)


    print("Level set")
    c = Control(s)
    Jhat = LevelSetLagrangian(Jform, c, phi, derivative_cb_pre=deriv_cb, lagrange_multiplier=[4e5, 4e5], penalty_value=[1e1, 1e1], penalty_update=[2.0, 2.0], constraint=[Power1, Power2], method='AL')
    print("Jhat")
    Jhat_v = Jhat(phi)
    print("Initial cost function value {:.5f}".format(Jhat_v))
    dJ = Jhat.derivative()
    Jhat.optimize_tape()

    velocity = Function(S)
    bcs_vel_1 = DirichletBC(S, noslip, 1)
    bcs_vel_2 = DirichletBC(S, noslip, 2)
    bcs_vel_3 = DirichletBC(S, noslip, 3)
    bcs_vel_4 = DirichletBC(S, noslip, 4)
    bcs_vel_5 = DirichletBC(S, noslip, 5)
    bcs_vel = [bcs_vel_1, bcs_vel_2, bcs_vel_3, bcs_vel_4, bcs_vel_5]
    reg_solver = RegularizationSolver(S, mesh, beta=1e3, dx=dx, sim_domain=0, iterative=iterative, output_dir=output_dir)

    hmin = 0.00940 # Hard coded from FEniCS

    options = {
             'hmin' : 0.00940,
             'hj_stab': 1.5,
             'dt_scale' : 1.0,
             'n_hj_steps' : 1,
             'n_reinit' : 10,
             'max_iter' : 60
             }
    parameters = {
        "mat_type" : "aij",
        "ksp_type" : "preonly",
        "ksp_monitor_true_residual": None,
        #"ksp_converged_reason" : None,
        "pc_type" : "lu",
        "pc_factor_mat_solver_type" : "mumps"
        }
    opti_solver = AugmentedLagrangianOptimization(Jhat, reg_solver, options=options)
    Jarr = opti_solver.solve(phi, velocity, solver_parameters=parameters, tolerance=1e-4)


if __name__ == '__main__':
    main()
