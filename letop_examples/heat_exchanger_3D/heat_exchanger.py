import firedrake as fd
from firedrake import inner, dot, grad, div, dx, ds, sin, pi, as_vector, cos
import firedrake_adjoint as fda

from letop.levelset import LevelSetFunctional, RegularizationSolver
from letop.optimization import (
    InfDimProblem,
    Constraint,
    nlspace_solve,
    read_checkpoint,
    is_checkpoint,
)
from letop.physics import (
    mark_no_flow_regions,
)
from physics_solvers import (
    temperature_solver,
    flow_solver,
    bc_velocity_profile,
)
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
    ns_solver_parameters,
)
from parameters_box import WALLS, DESIGN_DOMAIN

petsc4py.PETSc.Sys.popErrorHandler()


def print(x):
    return fd.PETSc.Sys.Print(x)


def forward(
    W,
    T,
    phi,
    opts,
    brinkmann_penalty,
    *,
    no_flow_domain_1,
    no_flow_domain_2,
    markers,
):
    # Parameters
    nu = fd.Constant(opts.nu)  # viscosity
    beta_gls = 0.9
    PeInv_val = 2e-4
    PeInv = fd.Constant(PeInv_val)
    t1 = fd.Constant(10.0)
    t2 = fd.Constant(1.0)

    x, y, z = fd.SpatialCoordinate(W.ufl_domain())

    inflow1, inflow2 = bc_velocity_profile(opts.type_he, (x, y, z))

    # Navier-Stokes 1
    w_sol1 = fd.Function(W)

    solver1 = flow_solver(
        W,
        w_sol1,
        no_flow=no_flow_domain_1,
        phi=phi,
        beta_gls=beta_gls,
        solver_parameters=ns_solver_parameters,
        brinkmann_penalty=brinkmann_penalty,
        design_domain=DESIGN_DOMAIN,
        inflow=inflow1,
        inlet=markers["INLET1"],
        walls=(markers["WALLS"], markers["INLET2"], markers["OUTLET2"]),
        nu=nu,
    )
    solver1.solve()
    u_sol1, _ = fd.split(w_sol1)

    # Navier-Stokes 2
    w_sol2 = fd.Function(W)
    solver2 = flow_solver(
        W,
        w_sol2,
        no_flow=no_flow_domain_2,
        phi=-phi,
        beta_gls=beta_gls,
        solver_parameters=ns_solver_parameters,
        brinkmann_penalty=brinkmann_penalty,
        design_domain=DESIGN_DOMAIN,
        inflow=inflow2,
        inlet=markers["INLET2"],
        walls=(markers["WALLS"], markers["INLET1"], markers["OUTLET1"]),
        nu=nu,
    )
    solver2.solve()
    u_sol2, _ = fd.split(w_sol2)

    beta = u_sol1 + u_sol2
    t = fd.Function(T, name="Temperature")
    solver_T = temperature_solver(
        T,
        t,
        beta,
        PeInv,
        solver_parameters=temperature_solver_parameters,
        t1=t1,
        INLET1=markers["INLET1"],
        t2=t2,
        INLET2=markers["INLET2"],
    )
    solver_T.solve()

    return w_sol1, w_sol2, t


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

    beta_param = 0.4
    mesh = fd.Mesh("./box_heat_exch.msh")

    from parameters_box import (
        INLET2,
        INLET1,
        OUTLET1,
        OUTLET2,
        INMOUTH1,
        INMOUTH2,
        OUTMOUTH1,
        OUTMOUTH2,
    )

    if opts.type_he == "counter":
        INLET1, OUTLET1 = OUTLET1, INLET1
    elif opts.type_he == "u_flow":
        INLET2, OUTLET1 = OUTLET1, INLET2
        OUTMOUTH1, INMOUTH2 = INMOUTH2, OUTMOUTH1

    markers = {
        "WALLS": WALLS,
        "INLET1": INLET1,
        "INLET2": INLET2,
        "OUTLET1": OUTLET1,
        "OUTLET2": OUTLET2,
    }

    no_flow_domain_1 = [INMOUTH2, OUTMOUTH2]
    no_flow_domain_2 = [INMOUTH1, OUTMOUTH1]
    no_flow = no_flow_domain_1.copy()
    no_flow.extend(no_flow_domain_2)
    mesh = mark_no_flow_regions(mesh, no_flow, no_flow)

    mh = fd.MeshHierarchy(mesh, opts.refinement)
    mesh = mh[-1]

    pressure_drop_constraint = opts.pressure_drop_constraint
    pressure_drop_1 = pressure_drop_constraint
    pressure_drop_2 = pressure_drop_constraint

    S = fd.VectorFunctionSpace(mesh, "CG", 1)

    PHI = fd.FunctionSpace(mesh, "CG", 1)
    phi = fd.Function(PHI, name="LevelSet")
    x, y, z = fd.SpatialCoordinate(mesh)

    ω = 0.25
    phi_expr = sin(y * pi / ω) * cos(x * pi / ω) * sin(
        z * pi / ω
    ) - fd.Constant(0.2)

    checkpoints = is_checkpoint(opts.output_dir)
    if checkpoints:
        current_iter = read_checkpoint(checkpoints, phi)
        with open(
            f"{opts.output_dir}/brinkmann_penalty.txt", "r"
        ) as txt_brinkmann:
            brinkmann_penalty_initial = fd.Constant(txt_brinkmann.read())
        print(
            f"Current brinkmann term: {brinkmann_penalty_initial.values()[0]}"
        )
    else:
        with stop_annotating():
            phi.interpolate(phi_expr)
        current_iter = 0
        brinkmann_penalty_initial = fd.Constant(opts.brinkmann_penalty)

    P2 = fd.VectorElement("CG", mesh.ufl_cell(), 1)
    P1 = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = fd.FunctionSpace(mesh, TH)
    print(f"DOFS: {W.dim()}")

    T = fd.FunctionSpace(mesh, "CG", 1)

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
        # w_sol1, w_sol2, t = forward(brinkmann_penalty)

        w_sol1, w_sol2, t = forward(
            W,
            T,
            phi,
            opts,
            brinkmann_penalty,
            no_flow_domain_1=no_flow_domain_1,
            no_flow_domain_2=no_flow_domain_2,
            markers=markers,
        )

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
