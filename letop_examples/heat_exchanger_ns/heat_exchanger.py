import firedrake as fd
import firedrake_adjoint as fda
from firedrake import inner, grad, ds, dx, sin, cos, pi, dot, div

from letop.levelset import LevelSetFunctional, RegularizationSolver
from letop.optimization import Constraint, InfDimProblem, nullspace_shape
from letop.physics import (
    NavierStokesBrinkmannForm,
    mark_no_flow_regions,
    InteriorBC,
    hs,
)
from pyadjoint import get_working_tape

from params import (
    line_sep,
    dist_center,
    inlet_width,
    WALLS,
    INLET1,
    INLET2,
    OUTLET1,
    OUTLET2,
    INMOUTH1,
    INMOUTH2,
    OUTMOUTH1,
    OUTMOUTH2,
    DOMAIN,
)

from pyadjoint import stop_annotating
import argparse


def heat_exchanger_navier_stokes():

    parser = argparse.ArgumentParser(description="Level set method parameters")
    parser.add_argument(
        "--nu",
        action="store",
        dest="nu",
        type=float,
        help="Viscosity",
        default=0.5,
    )
    opts, unknown = parser.parse_known_args()

    output_dir = "2D/"

    mesh = fd.Mesh("./2D_mesh.msh")
    no_flow_domain_1 = [3, 5]
    no_flow_domain_1_markers = [7, 8]
    no_flow_domain_2 = [2, 4]
    no_flow_domain_2_markers = [9, 10]
    no_flow = no_flow_domain_1.copy()
    no_flow.extend(no_flow_domain_2)
    no_flow_markers = no_flow_domain_1_markers.copy()
    no_flow_markers.extend(no_flow_domain_2_markers)
    mesh = mark_no_flow_regions(mesh, no_flow, no_flow_markers)

    S = fd.VectorFunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    PHI = fd.FunctionSpace(mesh, "CG", 1)
    phi_expr = sin(y * pi / 0.2) * cos(x * pi / 0.2) - fd.Constant(0.8)
    with stop_annotating():
        phi = fd.interpolate(phi_expr, PHI)
        phi.rename("LevelSet")
        fd.File(output_dir + "phi_initial.pvd").write(phi)

    # Parameters
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "ksp_converged_reason": None,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        # "snes_monitor" : None,
        # "snes_type" : "ksponly",
        # "snes_no_convergence_test" : None,
        # "snes_max_it": 1,
    }
    u_inflow = 1.0
    nu = fd.Constant(opts.nu)
    brinkmann_penalty = fd.Constant(1e5)

    P2 = fd.VectorElement("CG", mesh.ufl_cell(), 1)
    P1 = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = fd.FunctionSpace(mesh, TH)
    n = fd.FacetNormal(mesh)

    def forward(brinkmann_penalty):

        # Stokes 1
        w_sol1 = fd.Function(W)
        F1 = NavierStokesBrinkmannForm(
            W,
            w_sol1,
            nu,
            phi=-phi,
            brinkmann_penalty=brinkmann_penalty,
            design_domain=DOMAIN,
        )

        # Dirichelt boundary conditions
        inflow1 = fd.as_vector(
            [
                u_inflow
                * sin(
                    ((y - (line_sep - (dist_center + inlet_width))) * pi)
                    / inlet_width
                ),
                0.0,
            ]
        )

        noslip = fd.Constant((0.0, 0.0))

        bcs1_1 = fd.DirichletBC(W.sub(0), noslip, WALLS)
        bcs1_2 = fd.DirichletBC(W.sub(0), inflow1, INLET1)
        bcs1_3 = fd.DirichletBC(W.sub(1), fd.Constant(0.0), OUTLET1)
        bcs1_4 = fd.DirichletBC(W.sub(0), noslip, INLET2)
        bcs1_5 = fd.DirichletBC(W.sub(0), noslip, OUTLET2)
        bc_no_flow_1 = InteriorBC(W.sub(0), noslip, no_flow_domain_1_markers)
        bcs1 = [bcs1_1, bcs1_2, bcs1_3, bcs1_4, bcs1_5, bc_no_flow_1]
        # bcs1 = [bcs1_1, bcs1_2, bcs1_3, bcs1_4, bcs1_5]

        # Stokes 2
        w_sol2 = fd.Function(W)
        F2 = NavierStokesBrinkmannForm(
            W,
            w_sol2,
            nu,
            phi=phi,
            brinkmann_penalty=brinkmann_penalty,
            design_domain=DOMAIN,
        )
        inflow2 = fd.as_vector(
            [
                u_inflow
                * sin(((y - (line_sep + dist_center)) * pi) / inlet_width),
                0.0,
            ]
        )
        bcs2_1 = fd.DirichletBC(W.sub(0), noslip, WALLS)
        bcs2_2 = fd.DirichletBC(W.sub(0), inflow2, INLET2)
        bcs2_3 = fd.DirichletBC(W.sub(1), fd.Constant(0.0), OUTLET2)
        bcs2_4 = fd.DirichletBC(W.sub(0), noslip, INLET1)
        bcs2_5 = fd.DirichletBC(W.sub(0), noslip, OUTLET1)
        bc_no_flow_2 = InteriorBC(W.sub(0), noslip, no_flow_domain_2_markers)
        bcs2 = [bcs2_1, bcs2_2, bcs2_3, bcs2_4, bcs2_5, bc_no_flow_2]

        # Forward problems
        problem1 = fd.NonlinearVariationalProblem(F1, w_sol1, bcs=bcs1)
        problem2 = fd.NonlinearVariationalProblem(F2, w_sol2, bcs=bcs2)
        solver1 = fd.NonlinearVariationalSolver(
            problem1,
            solver_parameters=solver_parameters,
            options_prefix="ns_stokes1",
        )
        solver1.solve()
        solver2 = fd.NonlinearVariationalSolver(
            problem2,
            solver_parameters=solver_parameters,
            options_prefix="ns_stokes2",
        )
        solver2.solve()
        u1, p1 = fd.split(w_sol1)
        u2, p2 = fd.split(w_sol2)

        # Convection difussion equation
        k = fd.Constant(1e-3)
        t1 = fd.Constant(1.0)
        t2 = fd.Constant(10.0)

        h = fd.CellDiameter(mesh)
        T = fd.FunctionSpace(mesh, "CG", 1)
        t, rho = fd.Function(T), fd.TestFunction(T)
        n = fd.FacetNormal(mesh)
        beta = u1 + u2
        F = (
            inner(beta, grad(t)) * rho + k * inner(grad(t), grad(rho))
        ) * dx - inner(k * grad(t), n) * rho * (ds(OUTLET1) + ds(OUTLET2))

        R_U = dot(beta, grad(t)) - k * div(grad(t))
        beta_gls = 0.9
        h = fd.CellSize(mesh)
        tau_gls = beta_gls * (
            (4.0 * dot(beta, beta) / h ** 2) + 9.0 * (4.0 * k / h ** 2) ** 2
        ) ** (-0.5)
        degree = 4

        theta_U = dot(beta, grad(rho)) - k * div(grad(rho))
        F_T = F + tau_gls * inner(R_U, theta_U) * dx(degree=degree)

        bc1 = fd.DirichletBC(T, t1, INLET1)
        bc2 = fd.DirichletBC(T, t2, INLET2)
        bcs = [bc1, bc2]
        problem_T = fd.NonlinearVariationalProblem(F_T, t, bcs=bcs)
        solver_T = fd.NonlinearVariationalSolver(
            problem_T,
            solver_parameters=solver_parameters,
            options_prefix="temperature",
        )
        solver_T.solve()
        t.rename("Temperature")

        fd.File("temperature.pvd").write(t)

        return w_sol1, w_sol2, t

    def opti_problem(w_sol1, w_sol2, t):
        u1, p1 = fd.split(w_sol1)
        u2, p2 = fd.split(w_sol2)

        power_drop = 20
        Power1 = fd.assemble(p1 / power_drop * ds(INLET1))
        Power2 = fd.assemble(p2 / power_drop * ds(INLET2))
        scale_factor = 1.0
        Jform = fd.assemble(
            fd.Constant(-scale_factor) * inner(t * u1, n) * ds(OUTLET1)
        )
        return Jform, Power1, Power2

    # Plotting files and functions
    phi_pvd = fd.File("phi_evolution.pvd")
    flow_pvd = fd.File("flow_opti.pvd")

    w_pvd_1 = fd.Function(W)
    w_pvd_2 = fd.Function(W)

    def termination_event_1():
        p1_constraint = P1control.tape_value() - 1
        p2_constraint = P2control.tape_value() - 1
        event_value = max(p1_constraint, p2_constraint) - 1
        print(f"Value event: {event_value}")
        return event_value

    for termination_event, brinkmann_penalty in zip(
        [termination_event_1, None], [fd.Constant(1e4), fd.Constant(1e5)]
    ):
        s = fd.Function(S, name="deform")
        mesh.coordinates.assign(mesh.coordinates + s)
        w_sol1, w_sol2, t = forward(brinkmann_penalty)
        w_sol1_control = fda.Control(w_sol1)
        w_sol2_control = fda.Control(w_sol2)

        Jform, Power1, Power2 = opti_problem(w_sol1, w_sol2, t)

        def deriv_cb(phi):
            with stop_annotating():
                phi_pvd.write(phi[0])
                w_pvd_1.assign(w_sol1_control.tape_value())
                u1, p1 = w_pvd_1.split()
                u1.rename("vel1")
                w_pvd_2.assign(w_sol2_control.tape_value())
                u2, p2 = w_pvd_2.split()
                u2.rename("vel2")
                flow_pvd.write(u1, u2)

        c = fda.Control(s)

        # Reduced Functionals
        Jhat = LevelSetFunctional(Jform, c, phi, derivative_cb_pre=deriv_cb)
        P1hat = LevelSetFunctional(Power1, c, phi)
        P1control = fda.Control(Power1)

        P2hat = LevelSetFunctional(Power2, c, phi)
        P2control = fda.Control(Power2)

        print("Power drop 1 {:.5f}".format(Power1), flush=True)
        print("Power drop 2 {:.5f}".format(Power2), flush=True)

        beta_param = 0.08
        reg_solver = RegularizationSolver(
            S, mesh, beta=beta_param, gamma=1e5, dx=dx, design_domain=DOMAIN
        )

        tol = 1e-5
        dt = 0.05
        params = {
            "alphaC": 1.0,
            "debug": 5,
            "alphaJ": 1.0,
            "dt": dt,
            "K": 1e-3,
            "maxit": 200,
            # "maxit": 2,
            "maxtrials": 5,
            "itnormalisation": 10,
            "tol_merit": 5e-3,  # new merit can be within 0.5% of the previous merit
            # "normalize_tol" : -1,
            "tol": tol,
            "monitor_time": True,
        }

        problem = InfDimProblem(
            Jhat,
            reg_solver,
            ineqconstraints=[
                Constraint(P1hat, 1.0, P1control),
                Constraint(P2hat, 1.0, P2control),
            ],
            reinit_distance=0.1,
            termination_event=termination_event,
        )
        _ = nullspace_shape(problem, params)
        get_working_tape().clear_tape()


if __name__ == "__main__":
    heat_exchanger_navier_stokes()
