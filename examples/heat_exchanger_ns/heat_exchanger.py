import firedrake as fd
import firedrake_adjoint as fda
from firedrake import inner, grad, ds, dx, sin, cos, pi, dot, div

from lestofire.levelset import LevelSetFunctional, RegularizationSolver
from lestofire.optimization import Constraint, InfDimProblem
from lestofire.physics import NavierStokesBrinkmannForm
from nullspace_optimizer.lestofire import nlspace_solve_shape

from functools import partial

from params import (
    line_sep,
    dist_center,
    inlet_width,
    WALLS,
    INLET1,
    INLET2,
    OUTLET1,
    OUTLET2,
)

from pyadjoint import stop_annotating
import argparse


def heat_exchanger_navier_stokes():

    parser = argparse.ArgumentParser(description="Level set method parameters")
    parser.add_argument(
        "--nu", action="store", dest="nu", type=float, help="Viscosity", default=0.5
    )
    parser.add_argument(
        "--n_iters",
        dest="n_iters",
        type=int,
        action="store",
        default=1000,
        help="Number of optimization iterations",
    )
    opts, unknown = parser.parse_known_args()

    output_dir = "2D/"

    mesh = fd.Mesh("./2D_mesh.msh")

    S = fd.VectorFunctionSpace(mesh, "CG", 1)
    s = fd.Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    x, y = fd.SpatialCoordinate(mesh)
    PHI = fd.FunctionSpace(mesh, "DG", 1)
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
    }
    u_inflow = 1.0
    nu = fd.Constant(opts.nu)
    brinkmann_penalty = 1e4

    P2 = fd.VectorElement("CG", mesh.ufl_cell(), 1)
    P1 = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = fd.FunctionSpace(mesh, TH)

    # Stokes 1
    w_sol1 = fd.Function(W)
    F1 = NavierStokesBrinkmannForm(
        W,
        w_sol1,
        -phi,
        nu,
        brinkmann_penalty=brinkmann_penalty,
        design_domain=0,
        no_flow_domain=[3, 5],
    )

    # Dirichelt boundary conditions
    inflow1 = fd.as_vector(
        [
            u_inflow
            * sin(((y - (line_sep - (dist_center + inlet_width))) * pi) / inlet_width),
            0.0,
        ]
    )

    noslip = fd.Constant((0.0, 0.0))

    bcs1_1 = fd.DirichletBC(W.sub(0), noslip, WALLS)
    bcs1_2 = fd.DirichletBC(W.sub(0), inflow1, INLET1)
    bcs1_3 = fd.DirichletBC(W.sub(1), fd.Constant(0.0), OUTLET1)
    bcs1_4 = fd.DirichletBC(W.sub(0), noslip, INLET2)
    bcs1_5 = fd.DirichletBC(W.sub(0), noslip, OUTLET2)
    bcs1 = [bcs1_1, bcs1_2, bcs1_3, bcs1_4, bcs1_5]

    # Stokes 2
    w_sol2 = fd.Function(W)
    F2 = NavierStokesBrinkmannForm(
        W,
        w_sol2,
        phi,
        nu,
        brinkmann_penalty=brinkmann_penalty,
        design_domain=0,
        no_flow_domain=[2, 4],
    )
    inflow2 = fd.as_vector(
        [u_inflow * sin(((y - (line_sep + dist_center)) * pi) / inlet_width), 0.0]
    )
    bcs2_1 = fd.DirichletBC(W.sub(0), noslip, WALLS)
    bcs2_2 = fd.DirichletBC(W.sub(0), inflow2, INLET2)
    bcs2_3 = fd.DirichletBC(W.sub(1), fd.Constant(0.0), OUTLET2)
    bcs2_4 = fd.DirichletBC(W.sub(0), noslip, INLET1)
    bcs2_5 = fd.DirichletBC(W.sub(0), noslip, OUTLET1)
    bcs2 = [bcs2_1, bcs2_2, bcs2_3, bcs2_4, bcs2_5]

    # Forward problems
    problem1 = fd.NonlinearVariationalProblem(F1, w_sol1, bcs=bcs1)
    problem2 = fd.NonlinearVariationalProblem(F2, w_sol2, bcs=bcs2)
    solver1 = fd.NonlinearVariationalSolver(
        problem1, solver_parameters=solver_parameters
    )
    solver1.solve()
    w_sol1_control = fda.Control(w_sol1)
    solver2 = fd.NonlinearVariationalSolver(
        problem2, solver_parameters=solver_parameters
    )
    solver2.solve()
    w_sol2_control = fda.Control(w_sol2)

    # Convection difussion equation
    k = fd.Constant(1e-3)
    cp_value = 5.0e5
    t1 = fd.Constant(1.0)
    t2 = fd.Constant(10.0)

    # Mesh-related functions
    u1, p1 = fd.split(w_sol1)
    u2, p2 = fd.split(w_sol2)

    n = fd.FacetNormal(mesh)
    h = fd.CellDiameter(mesh)
    T = fd.FunctionSpace(mesh, "CG", 1)
    t, rho = fd.Function(T), fd.TestFunction(T)
    n = fd.FacetNormal(mesh)
    beta = u1 + u2
    F = (inner(beta, grad(t)) * rho + k * inner(grad(t), grad(rho))) * dx - inner(
        k * grad(t), n
    ) * rho * (ds(OUTLET1) + ds(OUTLET2))

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
    solver_parameters = {
        "ksp_type": "fgmres",
        "snes_atol": 1e-7,
        "pc_type": "hypre",
        "pc_hypre_type": "euclid",
        "ksp_max_it": 300,
    }
    solver_T = fd.NonlinearVariationalSolver(
        problem_T, solver_parameters=solver_parameters
    )
    solver_T.solve()
    t.rename("Temperature")

    fd.File("temperature.pvd").write(t)

    power_drop = 3e1
    Power1 = fd.assemble(p1 / power_drop * ds(INLET1))
    Power2 = fd.assemble(p2 / power_drop * ds(INLET2))
    scale_factor = 4e-5
    Jform = fd.assemble(
        fd.Constant(-scale_factor * cp_value) * inner(t * u1, n) * ds(OUTLET1)
    )

    phi_pvd = fd.File("phi_evolution.pvd")

    flow_pvd = fd.File("flow_opti.pvd")

    w_pvd_1 = fd.Function(W)
    w_pvd_2 = fd.Function(W)

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

    Jhat_v = Jhat(phi)
    print("Initial cost function value {:.5f}".format(Jhat_v), flush=True)
    print("Power drop 1 {:.5f}".format(Power1), flush=True)
    print("Power drop 2 {:.5f}".format(Power2), flush=True)

    beta_param = 0.08
    reg_solver = RegularizationSolver(
        S, mesh, beta=beta_param, gamma=1e5, dx=dx, design_domain=0
    )

    tol = 1e-5
    dt = 0.005
    params = {
        "alphaC": 1.0,
        "debug": 5,
        "alphaJ": 1.0,
        "dt": dt,
        "K": 1e-3,
        "maxit": opts.n_iters,
        # "maxit": 2,
        "maxtrials": 5,
        "itnormalisation": 10,
        "tol_merit": 5e-3,  # new merit can be within 0.5% of the previous merit
        # "normalize_tol" : -1,
        "tol": tol,
    }

    problem = InfDimProblem(
        Jhat,
        reg_solver,
        ineqconstraints=[
            Constraint(P1hat, 1.0, P1control),
            Constraint(P2hat, 1.0, P2control),
        ],
        reinit_steps=1,
    )
    _ = nlspace_solve_shape(problem, params)


if __name__ == "__main__":
    heat_exchanger_navier_stokes()
