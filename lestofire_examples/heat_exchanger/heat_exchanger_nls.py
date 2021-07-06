import firedrake as fd
import firedrake_adjoint as fda
from firedrake import (
    inner,
    derivative,
    grad,
    div,
    dx,
    ds,
    dot,
    dS,
    jump,
    avg,
    sin,
    cos,
    pi,
    exp,
)

from lestofire.levelset import LevelSetFunctional, RegularizationSolver
from lestofire.optimization import InfDimProblem, Constraint
from nullspace_optimizer.lestofire import nlspace_solve_shape

from pyadjoint import stop_annotating
import os

# Parameters
height = 1.5
shift_center = 0.52
dist_center = 0.03
inlet_width = 0.2
inlet_depth = 0.2
width = 1.2
line_sep = height / 2.0 - shift_center
# Line separating both inlets
DOMAIN = 0
INMOUTH1 = 2
INMOUTH2 = 3
OUTMOUTH1 = 4
OUTMOUTH2 = 5
INLET1, OUTLET1, INLET2, OUTLET2, WALLS, DESIGNBC = 1, 2, 3, 4, 5, 6
ymin1 = line_sep - (dist_center + inlet_width)
ymin2 = line_sep + dist_center


def heat_exchanger_optimization(mu=0.03, n_iters=1000):

    output_dir = "2D/"

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    mesh = fd.Mesh(f"{dir_path}/2D_mesh.msh")
    # Perturb the mesh coordinates. Necessary to calculate shape derivatives
    S = fd.VectorFunctionSpace(mesh, "CG", 1)
    s = fd.Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    # Initial level set function
    x, y = fd.SpatialCoordinate(mesh)
    PHI = fd.FunctionSpace(mesh, "CG", 1)
    phi_expr = sin(y * pi / 0.2) * cos(x * pi / 0.2) - fd.Constant(0.8)
    # Avoid recording the operation interpolate into the tape.
    # Otherwise, the shape derivatives will not be correct
    with fda.stop_annotating():
        phi = fd.interpolate(phi_expr, PHI)
        phi.rename("LevelSet")
        fd.File(output_dir + "phi_initial.pvd").write(phi)

    # Physics
    mu = fd.Constant(mu)  # viscosity
    alphamin = 1e-12
    alphamax = 2.5 / (2e-4)
    parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "ksp_converged_reason": None,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    stokes_parameters = parameters
    temperature_parameters = parameters
    u_inflow = 2e-3
    tin1 = fd.Constant(10.0)
    tin2 = fd.Constant(100.0)

    P2 = fd.VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = fd.FunctionSpace(mesh, TH)

    U = fd.TrialFunction(W)
    u, p = fd.split(U)
    V = fd.TestFunction(W)
    v, q = fd.split(V)

    epsilon = fd.Constant(10000.0)

    def hs(phi, epsilon):
        return fd.Constant(alphamax) * fd.Constant(1.0) / (
            fd.Constant(1.0) + exp(-epsilon * phi)
        ) + fd.Constant(alphamin)

    def stokes(phi, BLOCK_INLET_MOUTH, BLOCK_OUTLET_MOUTH):
        a_fluid = mu * inner(grad(u), grad(v)) - div(v) * p - q * div(u)
        darcy_term = inner(u, v)
        return (
            a_fluid * dx
            + hs(phi, epsilon) * darcy_term * dx(0)
            + alphamax
            * darcy_term
            * (dx(BLOCK_INLET_MOUTH) + dx(BLOCK_OUTLET_MOUTH))
        )

    # Dirichlet boundary conditions
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
    inflow2 = fd.as_vector(
        [
            u_inflow
            * sin(((y - (line_sep + dist_center)) * pi) / inlet_width),
            0.0,
        ]
    )

    noslip = fd.Constant((0.0, 0.0))

    # Stokes 1
    bcs1_1 = fd.DirichletBC(W.sub(0), noslip, WALLS)
    bcs1_2 = fd.DirichletBC(W.sub(0), inflow1, INLET1)
    bcs1_3 = fd.DirichletBC(W.sub(1), fd.Constant(0.0), OUTLET1)
    bcs1_4 = fd.DirichletBC(W.sub(0), noslip, INLET2)
    bcs1_5 = fd.DirichletBC(W.sub(0), noslip, OUTLET2)
    bcs1 = [bcs1_1, bcs1_2, bcs1_3, bcs1_4, bcs1_5]

    # Stokes 2
    bcs2_1 = fd.DirichletBC(W.sub(0), noslip, WALLS)
    bcs2_2 = fd.DirichletBC(W.sub(0), inflow2, INLET2)
    bcs2_3 = fd.DirichletBC(W.sub(1), fd.Constant(0.0), OUTLET2)
    bcs2_4 = fd.DirichletBC(W.sub(0), noslip, INLET1)
    bcs2_5 = fd.DirichletBC(W.sub(0), noslip, OUTLET1)
    bcs2 = [bcs2_1, bcs2_2, bcs2_3, bcs2_4, bcs2_5]

    # Forward problems
    U1, U2 = fd.Function(W), fd.Function(W)
    L = inner(fd.Constant((0.0, 0.0, 0.0)), V) * dx
    problem = fd.LinearVariationalProblem(
        stokes(-phi, INMOUTH2, OUTMOUTH2), L, U1, bcs=bcs1
    )
    solver_stokes1 = fd.LinearVariationalSolver(
        problem, solver_parameters=stokes_parameters, options_prefix="stokes_1"
    )
    solver_stokes1.solve()
    problem = fd.LinearVariationalProblem(
        stokes(phi, INMOUTH1, OUTMOUTH1), L, U2, bcs=bcs2
    )
    solver_stokes2 = fd.LinearVariationalSolver(
        problem, solver_parameters=stokes_parameters, options_prefix="stokes_2"
    )
    solver_stokes2.solve()

    # Convection difussion equation
    ks = fd.Constant(1e0)
    cp_value = 5.0e5
    cp = fd.Constant(cp_value)
    T = fd.FunctionSpace(mesh, "DG", 1)
    t = fd.Function(T, name="Temperature")
    w = fd.TestFunction(T)

    # Mesh-related functions
    n = fd.FacetNormal(mesh)
    h = fd.CellDiameter(mesh)
    u1, p1 = fd.split(U1)
    u2, p2 = fd.split(U2)

    def upwind(u):
        return (dot(u, n) + abs(dot(u, n))) / 2.0

    u1n = upwind(u1)
    u2n = upwind(u2)

    # Penalty term
    alpha = fd.Constant(500.0)
    # Bilinear form
    a_int = dot(grad(w), ks * grad(t) - cp * (u1 + u2) * t) * dx

    a_fac = (
        fd.Constant(-1.0) * ks * dot(avg(grad(w)), jump(t, n)) * dS
        + fd.Constant(-1.0) * ks * dot(jump(w, n), avg(grad(t))) * dS
        + ks("+") * (alpha("+") / avg(h)) * dot(jump(w, n), jump(t, n)) * dS
    )

    a_vel = (
        dot(
            jump(w),
            cp * (u1n("+") + u2n("+")) * t("+")
            - cp * (u1n("-") + u2n("-")) * t("-"),
        )
        * dS
        + dot(w, cp * (u1n + u2n) * t) * ds
    )

    a_bnd = (
        dot(w, cp * dot(u1 + u2, n) * t) * (ds(INLET1) + ds(INLET2))
        + w * t * (ds(INLET1) + ds(INLET2))
        - w * tin1 * ds(INLET1)
        - w * tin2 * ds(INLET2)
        + alpha / h * ks * w * t * (ds(INLET1) + ds(INLET2))
        - ks * dot(grad(w), t * n) * (ds(INLET1) + ds(INLET2))
        - ks * dot(grad(t), w * n) * (ds(INLET1) + ds(INLET2))
    )

    aT = a_int + a_fac + a_vel + a_bnd

    LT_bnd = (
        alpha / h * ks * tin1 * w * ds(INLET1)
        + alpha / h * ks * tin2 * w * ds(INLET2)
        - tin1 * ks * dot(grad(w), n) * ds(INLET1)
        - tin2 * ks * dot(grad(w), n) * ds(INLET2)
    )

    problem = fd.LinearVariationalProblem(derivative(aT, t), LT_bnd, t)
    solver_temp = fd.LinearVariationalSolver(
        problem,
        solver_parameters=temperature_parameters,
        options_prefix="temperature",
    )
    solver_temp.solve()
    # fd.solve(eT == 0, t, solver_parameters=temperature_parameters)

    # Cost function: Flux at the cold outlet
    scale_factor = 4e-4
    Jform = fd.assemble(
        fd.Constant(-scale_factor * cp_value) * inner(t * u1, n) * ds(OUTLET1)
    )
    # Constraints: Pressure drop on each fluid
    power_drop = 1e-2
    Power1 = fd.assemble(p1 / power_drop * ds(INLET1))
    Power2 = fd.assemble(p2 / power_drop * ds(INLET2))

    phi_pvd = fd.File("phi_evolution.pvd")

    def deriv_cb(phi):
        with stop_annotating():
            phi_pvd.write(phi[0])

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
    # Regularize the shape derivatives only in the domain marked with 0
    reg_solver = RegularizationSolver(
        S, mesh, beta=beta_param, gamma=1e5, dx=dx, design_domain=0
    )

    tol = 1e-5
    dt = 0.05
    params = {
        "alphaC": 1.0,
        "debug": 5,
        "alphaJ": 1.0,
        "dt": dt,
        "K": 1e-3,
        "maxit": n_iters,
        "maxtrials": 5,
        "itnormalisation": 10,
        "tol_merit": 5e-3,  # new merit can be within 0.5% of the previous merit
        # "normalize_tol" : -1,
        "tol": tol,
    }

    solver_parameters = {
        "reinit_solver": {
            "h_factor": 2.0,
        }
    }
    # Optimization problem
    problem = InfDimProblem(
        Jhat,
        reg_solver,
        ineqconstraints=[
            Constraint(P1hat, 1.0, P1control),
            Constraint(P2hat, 1.0, P2control),
        ],
        solver_parameters=solver_parameters,
    )
    results = nlspace_solve_shape(problem, params)

    return results


if __name__ == "__main__":
    heat_exchanger_optimization()
