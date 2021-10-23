import firedrake as fd
from firedrake import inner, dot, grad, div, dx, ds, sin, pi, as_vector, cos
import firedrake_adjoint as fda

from lestofire.levelset import LevelSetFunctional, RegularizationSolver
from lestofire.optimization import InfDimProblem, Constraint, nlspace_solve
from lestofire.physics import (
    NavierStokesBrinkmannForm,
    NavierStokesBrinkmannSolver,
    mark_no_flow_regions,
    InteriorBC,
)
import signal
import glob
import re
from itertools import count
import petsc4py
from pyadjoint import stop_annotating
import argparse
from copy import copy

petsc4py.PETSc.Sys.popErrorHandler()


def print(x):
    return fd.PETSc.Sys.Print(x)


def heat_exchanger_3D():
    parser = argparse.ArgumentParser(description="Level set method parameters")
    parser.add_argument(
        "--nu",
        action="store",
        dest="nu",
        type=float,
        help="Kinematic Viscosity",
        default=1.0,
    )
    parser.add_argument(
        "--hj_time",
        action="store",
        dest="hj_time",
        type=float,
        help="Total time for the HJ step",
        default=0.02,
    )
    parser.add_argument(
        "--brinkmann_penalty",
        action="store",
        dest="brinkmann_penalty",
        type=float,
        help="Brinkmann term",
        default=1e5,
    )
    parser.add_argument(
        "--K",
        action="store",
        dest="K",
        type=float,
        help="Feeling distance parameter",
        default=1e-3,
    )
    parser.add_argument(
        "--beta_param",
        action="store",
        dest="beta_param",
        type=float,
        help="Regularization parameter",
        default=0.1,
    )
    parser.add_argument(
        "--refinement",
        action="store",
        dest="refinement",
        type=int,
        help="Level of refinement",
        default=0,
    )
    parser.add_argument(
        "--alphaC",
        action="store",
        dest="alphaC",
        type=float,
        help="Weight for the constraints",
        default=0.1,
    )
    parser.add_argument(
        "--scale_factor",
        action="store",
        dest="scale_factor",
        type=float,
        help="Scale heat exchange cost function",
        default=5.0,
    )
    parser.add_argument(
        "--pressure_drop",
        action="store",
        dest="pressure_drop",
        type=float,
        help="Power drop constraint",
        default=10.0,
    )
    parser.add_argument(
        "--n_iters",
        dest="n_iters",
        type=int,
        action="store",
        default=1000,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        action="store",
        default="./",
        help="Output folder",
    )
    parser.add_argument(
        "--type",
        dest="type_he",
        type=str,
        action="store",
        default="parallel",
        help="Type of heat exchanger: parallel or counter",
    )
    opts = parser.parse_args()
    print(f"Parameters used: {opts}")

    beta_gls = 0.9
    mesh = fd.Mesh("./box_heat_exch.msh")

    from parameters_box import (
        WALLS,
        INLET2,
        INLET1,
        OUTLET1,
        OUTLET2,
        DESIGN_DOMAIN,
        INMOUTH1,
        INMOUTH2,
        OUTMOUTH1,
        OUTMOUTH2,
        inlet_1_coords,
        inlet_2_coords,
        outlet_1_coords,
        R1,
        R2,
    )

    if opts.type_he == "parallel":
        sign_inlet_1 = 1
        sign_inlet_2 = 1
    elif opts.type_he == "counter":
        INLET1, OUTLET1 = OUTLET1, INLET1
        inlet_1_coords, outlet_1_coords = outlet_1_coords, inlet_1_coords
        sign_inlet_1 = -1
        sign_inlet_2 = 1
    elif opts.type_he == "u_flow":
        INLET2, OUTLET1 = OUTLET1, INLET2
        inlet_2_coords, outlet_1_coords = outlet_1_coords, inlet_2_coords
        OUTMOUTH1, INMOUTH2 = INMOUTH2, OUTMOUTH1
        R1, R2 = R2, R1
        sign_inlet_1 = 1
        sign_inlet_2 = -1
    else:
        raise RuntimeError(
            f"type of heat exchanger {opts.type_he} is invalid. \
                                Choose parallel, counter or u_flow"
        )

    no_flow_domain_1 = [INMOUTH2, OUTMOUTH2]
    no_flow_domain_2 = [INMOUTH1, OUTMOUTH1]
    no_flow = no_flow_domain_1.copy()
    no_flow.extend(no_flow_domain_2)
    mesh = mark_no_flow_regions(mesh, no_flow, no_flow)

    mh = fd.MeshHierarchy(mesh, opts.refinement)
    mesh = mh[-1]

    # Parameters
    nu = fd.Constant(opts.nu)  # viscosity
    u_inflow = 1.0
    pressure_drop = opts.pressure_drop
    pressure_drop_1 = pressure_drop
    pressure_drop_2 = pressure_drop
    ns_solver_tolerances = 1e-4

    PeInv_val = 2e-4
    PeInv = fd.Constant(PeInv_val)
    t1 = fd.Constant(10.0)
    t2 = fd.Constant(1.0)

    S = fd.VectorFunctionSpace(mesh, "CG", 1)

    PHI = fd.FunctionSpace(mesh, "CG", 1)
    phi = fd.Function(PHI)
    x, y, z = fd.SpatialCoordinate(mesh)

    ω = 0.25
    phi_expr = sin(y * pi / ω) * cos(x * pi / ω) * sin(
        z * pi / ω
    ) - fd.Constant(0.2)

    with stop_annotating():
        checkpoints = glob.glob(f"{opts.output_dir}/checkpoint*")
        checkpoints_sorted = sorted(
            checkpoints,
            key=lambda L: list(map(int, re.findall(r"iter_(\d+)\.h5", L)))[0],
        )

        print(f"checkpoints: {checkpoints_sorted}")
        if checkpoints_sorted:
            last_file = checkpoints_sorted[-1]
            current_iter = int(re.findall(r"iter_(\d+)\.h5", last_file)[0])
            with fd.HDF5File(
                f"{opts.output_dir}/checkpoint_iter_{current_iter}.h5", "r"
            ) as checkpoint:
                checkpoint.read(phi, "/checkpoint")
            print(f"Restarting simulation at {current_iter}")
            with open(
                f"{opts.output_dir}/brinkmann_penalty.txt", "r"
            ) as txt_brinkmann:
                brinkmann_penalty_initial = fd.Constant(txt_brinkmann.read())
            print(
                f"Current brinkmann term: {brinkmann_penalty_initial.values()[0]}"
            )
        else:
            phi.interpolate(phi_expr)
            current_iter = 0
            brinkmann_penalty_initial = fd.Constant(opts.brinkmann_penalty)
        phi.rename("LevelSet")
        fd.File(f"{opts.output_dir}/phi_initial.pvd").write(phi)

    P2 = fd.VectorElement("CG", mesh.ufl_cell(), 1)
    P1 = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = fd.FunctionSpace(mesh, TH)
    n = fd.FacetNormal(mesh)
    print(f"DOFS: {W.dim()}")

    T = fd.FunctionSpace(mesh, "CG", 1)
    t, rho = fd.Function(T), fd.TestFunction(T)

    global_counter = count(current_iter)

    def receive_signal(signum, stack):
        iter_current = next(copy(global_counter))
        print(f"Received: {signum}, iter: {iter_current}")
        with fd.HDF5File(
            f"{opts.output_dir}/checkpoint_iter_{iter_current}.h5", "w"
        ) as checkpoint:
            checkpoint.write(phi, "/checkpoint")
        with open(
            f"{opts.output_dir}/brinkmann_penalty.txt", "w"
        ) as txt_brinkmann:
            txt_brinkmann.write(str(brinkmann_penalty.values()[0]))

    signal.signal(signal.SIGHUP, receive_signal)

    def forward(brinkmann_penalty):

        # Navier-Stokes 1
        w_sol1 = fd.Function(W)
        F = NavierStokesBrinkmannForm(
            W,
            w_sol1,
            nu,
            phi=phi,
            brinkmann_penalty=brinkmann_penalty,
            design_domain=DESIGN_DOMAIN,
            beta_gls=beta_gls,
        )

        x1, y1, z1 = inlet_1_coords
        r1_2 = (x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2
        inflow1 = as_vector(
            [sign_inlet_1 * u_inflow / R1 ** 2 * (R1 ** 2 - r1_2), 0.0, 0.0]
        )

        noslip = fd.Constant((0.0, 0.0, 0.0))
        bcs1_1 = fd.DirichletBC(W.sub(0), noslip, (WALLS, INLET2, OUTLET2))
        bcs1_2 = fd.DirichletBC(W.sub(0), inflow1, INLET1)
        bcs1_no_flow = InteriorBC(W.sub(0), noslip, no_flow_domain_1)
        bcs = [bcs1_1, bcs1_2, bcs1_no_flow]

        problem = fd.NonlinearVariationalProblem(F, w_sol1, bcs=bcs)

        solver_parameters = {
            "ksp_converged_maxits": None,
            "ksp_max_it": 1000,
            "ksp_atol": ns_solver_tolerances,
            "ksp_rtol": ns_solver_tolerances,
            # "ksp_monitor": None,
            "snes_atol": 1e-4,
            "snes_rtol": 1e-4,
            "snes_max_it": 10,
            "snes_no_convergence_test": None,
            "ksp_converged_reason": None,
        }
        solver = NavierStokesBrinkmannSolver(
            problem, solver_parameters=solver_parameters
        )
        solver.solve()
        u_sol1, p_sol1 = fd.split(w_sol1)

        # Navier-Stokes 2
        w_sol2 = fd.Function(W)
        F = NavierStokesBrinkmannForm(
            W,
            w_sol2,
            nu,
            phi=-phi,
            brinkmann_penalty=brinkmann_penalty,
            design_domain=DESIGN_DOMAIN,
            beta_gls=beta_gls,
        )
        x2, y2, z2 = inlet_2_coords
        r2_2 = (x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2
        inflow2 = as_vector(
            [sign_inlet_2 * u_inflow / R2 ** 2 * (R2 ** 2 - r2_2), 0.0, 0.0]
        )

        bcs2_1 = fd.DirichletBC(W.sub(0), noslip, (WALLS, INLET1, OUTLET1))
        bcs2_2 = fd.DirichletBC(W.sub(0), inflow2, INLET2)
        bcs2_no_flow = InteriorBC(W.sub(0), noslip, no_flow_domain_2)
        bcs_2 = [bcs2_1, bcs2_2, bcs2_no_flow]

        problem = fd.NonlinearVariationalProblem(F, w_sol2, bcs=bcs_2)

        solver2 = NavierStokesBrinkmannSolver(
            problem, solver_parameters=solver_parameters
        )
        solver2.solve()
        u_sol2, _ = fd.split(w_sol2)

        # Temperature advection diffusion equation
        beta = u_sol1 + u_sol2
        F = (
            inner(beta, grad(t)) * rho + PeInv * inner(grad(t), grad(rho))
        ) * dx - inner(PeInv * grad(t), n) * rho * (ds(OUTLET1) + ds(OUTLET2))

        R_U = dot(beta, grad(t)) - PeInv * div(grad(t))
        beta_gls_temp = 0.9
        h = fd.CellSize(mesh)
        tau_gls = beta_gls_temp * (
            (4.0 * dot(beta, beta) / h ** 2)
            + 9.0 * (4.0 * PeInv / h ** 2) ** 2
        ) ** (-0.5)
        degree = 4

        theta_U = dot(beta, grad(rho)) - PeInv * div(grad(rho))
        F_T = F + tau_gls * inner(R_U, theta_U) * dx(degree=degree)

        bc1 = fd.DirichletBC(T, t1, INLET1)
        bc2 = fd.DirichletBC(T, t2, INLET2)
        bcs = [bc1, bc2]
        problem_T = fd.NonlinearVariationalProblem(F_T, t, bcs=bcs)
        temperature_solver_parameters = {
            "ksp_type": "fgmres",
            "snes_atol": 1e-6,
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_mat_type": "aij",
            "assembled_pc_type": "hypre",
            "assembled_pc_hypre_boomeramg_P_max": 4,
            "assembled_pc_hypre_boomeramg_agg_nl": 1,
            "assembled_pc_hypre_boomeramg_agg_num_paths": 2,
            "assembled_pc_hypre_boomeramg_coarsen_type": "HMIS",
            "assembled_pc_hypre_boomeramg_interp_type": "ext+i",
            "assembled_pc_hypre_boomeramg_no_CF": True,
            "ksp_max_it": 300,
        }
        solver_T = fd.NonlinearVariationalSolver(
            problem_T, solver_parameters=temperature_solver_parameters
        )
        solver_T.solve()
        t.rename("Temperature")

        return w_sol1, w_sol2, t

    def opti_problem(w_sol1, w_sol2, t):
        u_sol1, p_sol1 = fd.split(w_sol1)
        u_sol2, p_sol2 = fd.split(w_sol2)

        Power1 = (
            fd.assemble(p_sol1 * ds(INLET1) - p_sol1 * ds(OUTLET1))
            / pressure_drop_1
        )
        Power2 = (
            fd.assemble(p_sol2 * ds(INLET2) - p_sol2 * ds(OUTLET2))
            / pressure_drop_2
        )

        scale_factor = opts.scale_factor
        J = fd.assemble(
            fd.Constant(-scale_factor) * inner(t * u_sol2, n) * ds(OUTLET2)
        )
        J_hot = fd.assemble(
            fd.Constant(-scale_factor) * inner(t * u_sol1, n) * ds(OUTLET1)
        )
        return J, J_hot, Power1, Power2

    phi_pvd = fd.File(
        f"{opts.output_dir}/phi_evolution.pvd",
        target_degree=1,
        target_continuity=fd.H1,
        mode="a",
    )
    ns1 = fd.File(f"{opts.output_dir}/navier_stokes_1.pvd", mode="a")
    ns2 = fd.File(f"{opts.output_dir}/navier_stokes_2.pvd", mode="a")
    temperature = fd.File(f"{opts.output_dir}/temperature.pvd", mode="a")
    temperature_pvd = fd.Function(T)

    def termination_event_1():
        p1_constraint = P1control.tape_value() - 1
        p2_constraint = P2control.tape_value() - 1
        event_value = max(p1_constraint, p2_constraint)
        print(f"Value event: {event_value}")
        return event_value

    def termination_event_2():
        iter_current = next(copy(global_counter))
        print(f"Value event iter count: {iter_current}")
        return float(iter_current % 500)

    brinkmann_penalty_initial_value = brinkmann_penalty_initial.values()[0]
    if brinkmann_penalty_initial_value > opts.brinkmann_penalty:
        termination_events = [termination_event_2, termination_event_2]
        brinkmann_pen_terms = [
            brinkmann_penalty_initial,
            fd.Constant(brinkmann_penalty_initial_value * 10),
        ]
    else:
        termination_events = [termination_event_1, None]
        brinkmann_pen_terms = [
            brinkmann_penalty_initial,
            fd.Constant(brinkmann_penalty_initial_value * 5),
        ]

    for termination_event, brinkmann_penalty in zip(
        termination_events, brinkmann_pen_terms
    ):

        s = fd.Function(S, name="deform")
        mesh.coordinates.assign(mesh.coordinates + s)
        w_sol1, w_sol2, t = forward(brinkmann_penalty)
        w_sol1_control = fda.Control(w_sol1)
        w_sol2_control = fda.Control(w_sol2)
        t_control = fda.Control(t)

        J, J_hot, Power1, Power2 = opti_problem(w_sol1, w_sol2, t)
        c = fda.Control(s)
        J_hot_control = fda.Control(J_hot)

        def deriv_cb(phi):
            with stop_annotating():
                iter = next(global_counter)
                print(f"Hot flux: {J_hot_control.tape_value()}")
                if iter % 15 == 0:
                    u_sol1, p_sol1 = w_sol1_control.tape_value().split()
                    u_sol2, p_sol2 = w_sol2_control.tape_value().split()

                    u_sol1.rename("Velocity1")
                    p_sol1.rename("Pressure1")

                    u_sol2.rename("Velocity2")
                    p_sol2.rename("Pressure2")

                    ns1.write(u_sol1, p_sol1, time=iter)
                    ns2.write(u_sol2, p_sol2, time=iter)
                    phi_pvd.write(phi[0], time=iter)

                    temperature_pvd.assign(t_control.tape_value())
                    temperature_pvd.rename("temperature")
                    temperature.write(temperature_pvd, time=iter)

        # Reduced Functionals
        Jhat = LevelSetFunctional(J, c, phi, derivative_cb_pre=deriv_cb)
        P1hat = LevelSetFunctional(Power1, c, phi)
        P1control = fda.Control(Power1)

        P2hat = LevelSetFunctional(Power2, c, phi)
        P2control = fda.Control(Power2)
        print("Power drop 1 {:.5f}".format(Power1))
        print("Power drop 2 {:.5f}".format(Power2))

        beta_param = opts.beta_param
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
            "ksp_converged_reason": None,
        }
        reg_solver = RegularizationSolver(
            S,
            mesh,
            beta=beta_param,
            gamma=1e6,
            dx=dx,
            design_domain=DESIGN_DOMAIN,
            solver_parameters=iterative_parameters,
        )

        tol = 1e-7
        dt = opts.hj_time
        params = {
            "alphaC": 0.1,
            "debug": 5,
            "alphaJ": 0.1,
            "dt": dt,
            "K": opts.K,
            "maxit": opts.n_iters,
            "maxtrials": 10,
            "itnormalisation": 10,
            "tol_merit": 1e-2,  # new merit can be within 1% of the previous merit
            # "normalize_tol" : -1,
            "tol": tol,
        }

        hj_solver_parameters = {
            "peclet_number": 1e-3,
            "ksp_type": "gmres",
            "ksp_converged_maxits": None,
            "snes_type": "ksponly",
            "snes_no_convergence_test": None,
            "snes_atol": 1e-5,
            "ksp_atol": 1e-5,
            "ksp_rtol": 1e-5,
            "ts_type": "beuler",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
            "sub_pc_factor_levels": 1,
            "ksp_max_it": 5000,
            "ksp_gmres_restart": 200,
            "ts_atol": 1e-5,
            "ts_rtol": 1e-5,
            "ts_dt": dt / 50.0,
            "ts_max_steps": 400,
            # "ksp_monitor": None
            # "snes_monitor": None,
            # "ts_monitor": None,
            # "ksp_converged_reason": None,
        }

        solver_parameters = {
            "hj_solver": hj_solver_parameters,
            "reinit_solver": {
                "ksp_type": "cg",
                "ksp_rtol": 1e-6,
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
                "pc_hypre_boomeramg_max_iter": 5,
                "pc_hypre_boomeramg_coarsen_type": "PMIS",
                "pc_hypre_boomeramg_agg_nl": 2,
                "pc_hypre_boomeramg_strong_threshold": 0.95,
                "pc_hypre_boomeramg_interp_type": "ext+i",
                "pc_hypre_boomeramg_P_max": 2,
                "pc_hypre_boomeramg_relax_type_all": "sequential-Gauss-Seidel",
                "pc_hypre_boomeramg_grid_sweeps_all": 1,
                "pc_hypre_boomeramg_truncfactor": 0.3,
                "pc_hypre_boomeramg_max_levels": 6,
                "ksp_max_it": 200,
                "ksp_converged_maxits": None,
            },
        }

        problem = InfDimProblem(
            Jhat,
            reg_solver,
            ineqconstraints=[
                Constraint(P1hat, 1.0, P1control),
                Constraint(P2hat, 1.0, P2control),
            ],
            reinit_distance=0.08,
            solver_parameters=solver_parameters,
        )
        problem.set_termination_event(
            termination_event, termination_tolerance=1e-1
        )

        _ = nlspace_solve(problem, params)
        fda.get_working_tape().clear_tape()


if __name__ == "__main__":
    fd.parameters["form_compiler"]["quadrature_degree"] = 4
    heat_exchanger_3D()
