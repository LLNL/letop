import firedrake as fd
from firedrake import inner, dot, grad, div, dx, ds, sin, pi, as_vector, cos
import firedrake_adjoint as fda

from lestofire.levelset import LevelSetFunctional, RegularizationSolver
from lestofire.optimization import (
    InfDimProblem,
    Constraint,
    nlspace_solve,
    read_checkpoint,
)
from lestofire.physics import (
    NavierStokesBrinkmannForm,
    NavierStokesBrinkmannSolver,
    mark_no_flow_regions,
    InteriorBC,
)
from physics_solvers import temperature_solver
import signal
from qois import pressure_drop, heat_flux
from itertools import count
import petsc4py
from pyadjoint import stop_annotating
import argparse
from copy import copy
from solver_parameters import (
    reinit_solver_parameters,
    hj_solver_parameters,
    regularization_solver_parameters,
    temperature_solver_parameters,
)

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
        "--brinkmann_penalty",
        action="store",
        dest="brinkmann_penalty",
        type=float,
        help="Brinkmann term",
        default=1e5,
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
        "--pressure_drop_constraint",
        action="store",
        dest="pressure_drop_constraint",
        type=float,
        help="Pressure drop constraint",
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
    beta_param = 0.4
    u_inflow = 1.0
    PeInv_val = 2e-4
    t1 = fd.Constant(10.0)
    t2 = fd.Constant(1.0)
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
    pressure_drop_constraint = opts.pressure_drop_constraint
    pressure_drop_1 = pressure_drop_constraint
    pressure_drop_2 = pressure_drop_constraint
    ns_solver_tolerances = 1e-4

    PeInv = fd.Constant(PeInv_val)

    S = fd.VectorFunctionSpace(mesh, "CG", 1)

    PHI = fd.FunctionSpace(mesh, "CG", 1)
    phi = fd.Function(PHI, name="LevelSet")
    x, y, z = fd.SpatialCoordinate(mesh)

    ω = 0.25
    phi_expr = sin(y * pi / ω) * cos(x * pi / ω) * sin(
        z * pi / ω
    ) - fd.Constant(0.2)

    if read_checkpoint(opts.output_dir, phi):
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

    P2 = fd.VectorElement("CG", mesh.ufl_cell(), 1)
    P1 = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = fd.FunctionSpace(mesh, TH)
    print(f"DOFS: {W.dim()}")

    T = fd.FunctionSpace(mesh, "CG", 1)
    t, _ = fd.Function(T), fd.TestFunction(T)

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

    def flow_solver(
        W, w_sol, nu, inflow, INLET, WALLS, no_flow, phi=None, beta_gls=0.9
    ):
        F = NavierStokesBrinkmannForm(
            W,
            w_sol,
            nu,
            phi=phi,
            brinkmann_penalty=brinkmann_penalty,
            design_domain=DESIGN_DOMAIN,
            beta_gls=beta_gls,
        )

        noslip = fd.Constant((0.0, 0.0, 0.0))
        bcs_1 = fd.DirichletBC(W.sub(0), noslip, WALLS)
        bcs_2 = fd.DirichletBC(W.sub(0), inflow, INLET)
        bcs_no_flow = InteriorBC(W.sub(0), noslip, no_flow)
        bcs = [bcs_1, bcs_2, bcs_no_flow]

        problem = fd.NonlinearVariationalProblem(F, w_sol, bcs=bcs)

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
        return NavierStokesBrinkmannSolver(
            problem, solver_parameters=solver_parameters
        )

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
        # solver = NavierStokesBrinkmannSolver(
        #    problem, solver_parameters=solver_parameters
        # )
        # solver.solve()
        u_sol1, p_sol1 = fd.split(w_sol1)
        solver_supp = flow_solver(
            W,
            w_sol1,
            nu,
            inflow1,
            INLET1,
            (WALLS, INLET2, OUTLET2),
            no_flow_domain_1,
            phi=phi,
            beta_gls=beta_gls,
        )
        solver_supp.solve()

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

        beta = u_sol1 + u_sol2
        solver_T = temperature_solver(
            T, t, beta, PeInv, t1, INLET1, t2, INLET2
        )
        solver_T.solve()
        t.rename("Temperature")

        return w_sol1, w_sol2, t

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

        J = heat_flux(w_sol2, t, OUTLET2)
        J_hot = heat_flux(w_sol1, t, OUTLET1)
        Pdrop1 = pressure_drop(w_sol1, INLET1, OUTLET1, pressure_drop_1)
        Pdrop2 = pressure_drop(w_sol2, INLET2, OUTLET2, pressure_drop_2)
        print(f"Cold flux: {J}, hot flux: {J_hot}")

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
        P1hat = LevelSetFunctional(Pdrop1, c, phi)
        P1control = fda.Control(Pdrop1)

        P2hat = LevelSetFunctional(Pdrop2, c, phi)
        P2control = fda.Control(Pdrop2)
        print("Pressure drop 1 {:.5f}".format(Pdrop1))
        print("Pressure drop 2 {:.5f}".format(Pdrop2))

        reg_solver = RegularizationSolver(
            S,
            mesh,
            beta=beta_param,
            gamma=1e6,
            dx=dx,
            design_domain=DESIGN_DOMAIN,
            solver_parameters=regularization_solver_parameters,
        )

        tol = 1e-7
        dt = 0.02
        params = {
            "alphaC": 0.1,
            "debug": 5,
            "alphaJ": 0.1,
            "dt": dt,
            "K": 1e-3,
            "maxit": opts.n_iters,
            "maxtrials": 10,
            "itnormalisation": 10,
            "tol_merit": 1e-2,  # new merit can be within 1% of the previous merit
            # "normalize_tol" : -1,
            "tol": tol,
        }

        hj_solver_parameters["ts_dt"] = dt / 50.0
        solver_parameters = {
            "hj_solver": hj_solver_parameters,
            "reinit_solver": reinit_solver_parameters,
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
