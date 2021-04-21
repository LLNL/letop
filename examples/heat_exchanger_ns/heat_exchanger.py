from firedrake import *
from firedrake_adjoint import *

from lestofire import (
    LevelSetFunctional,
    RegularizationSolver,
    HJLocalDG,
    ReinitSolverDG,
    NavierStokesBrinkmannForm,
    NavierStokesBrinkmannSolver,
)
from nullspace_optimizer.lestofire import nlspace_solve_shape, Constraint, InfDimProblem

from params import (
    INMOUTH2,
    INMOUTH1,
    line_sep,
    dist_center,
    inlet_width,
    WALLS,
    INLET1,
    INLET2,
    OUTLET1,
    OUTLET2,
    width,
    OUTMOUTH1,
    OUTMOUTH2,
)

from pyadjoint import no_annotations, stop_annotating
import argparse

parser = argparse.ArgumentParser(description="Level set method parameters")
parser.add_argument(
    "--mu", action="store", dest="mu", type=float, help="Viscosity", default=0.03
)
parser.add_argument(
    "--n_iters",
    dest="n_iters",
    type=int,
    action="store",
    default=1000,
    help="Number of optimization iterations",
)
opts = parser.parse_args()

output_dir = "2D/"

mesh = Mesh("./2D_mesh.msh")

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="deform")
mesh.coordinates.assign(mesh.coordinates + s)

x, y = SpatialCoordinate(mesh)
PHI = FunctionSpace(mesh, "DG", 1)
phi_expr = sin(y * pi / 0.2) * cos(x * pi / 0.2) - Constant(0.8)
with stop_annotating():
    phi = interpolate(phi_expr, PHI)
    phi.rename("LevelSet")
    File(output_dir + "phi_initial.pvd").write(phi)


# Parameters
mu = Constant(opts.mu)  # viscosity
alphamin = 1e-12
alphamax = 2.5 / (2e-4)
solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "ksp_converged_reason": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
stokes_parameters = parameters
temperature_parameters = parameters
u_inflow = 1.0
tin1 = Constant(10.0)
tin2 = Constant(100.0)
nu = Constant(1.0)
brinkmann_penalty = 1e4

Re = u_inflow * width / mu.values()[0]


P2 = VectorElement("CG", mesh.ufl_cell(), 1)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)


# Stokes 1
w_sol1 = Function(W)
F1 = NavierStokesBrinkmannForm(
    W, w_sol1, phi, nu, brinkmann_penalty=brinkmann_penalty, design_domain=0
)


# Dirichelt boundary conditions
inflow1 = as_vector(
    [
        u_inflow
        * sin(((y - (line_sep - (dist_center + inlet_width))) * pi) / inlet_width),
        0.0,
    ]
)

noslip = Constant((0.0, 0.0))

bcs1_1 = DirichletBC(W.sub(0), noslip, WALLS)
bcs1_2 = DirichletBC(W.sub(0), inflow1, INLET1)
bcs1_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET1)
bcs1_4 = DirichletBC(W.sub(0), noslip, INLET2)
bcs1_5 = DirichletBC(W.sub(0), noslip, OUTLET2)
bcs1 = [bcs1_1, bcs1_2, bcs1_3, bcs1_4, bcs1_5]

# Stokes 2
w_sol2 = Function(W)
F2 = NavierStokesBrinkmannForm(
    W, w_sol2, -phi, nu, brinkmann_penalty=brinkmann_penalty, design_domain=0
)
inflow2 = as_vector(
    [u_inflow * sin(((y - (line_sep + dist_center)) * pi) / inlet_width), 0.0]
)
bcs2_1 = DirichletBC(W.sub(0), noslip, WALLS)
bcs2_2 = DirichletBC(W.sub(0), inflow2, INLET2)
bcs2_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET2)
bcs2_4 = DirichletBC(W.sub(0), noslip, INLET1)
bcs2_5 = DirichletBC(W.sub(0), noslip, OUTLET1)
bcs2 = [bcs2_1, bcs2_2, bcs2_3, bcs2_4, bcs2_5]

# Forward problems
problem1 = NonlinearVariationalProblem(F1, w_sol1, bcs=bcs1)
problem2 = NonlinearVariationalProblem(F2, w_sol2, bcs=bcs2)
solver1 = NonlinearVariationalSolver(problem1, solver_parameters=solver_parameters)
solver1.solve()
w_sol1_control = Control(w_sol1)
solver2 = NonlinearVariationalSolver(problem2, solver_parameters=solver_parameters)
solver2.solve()
w_sol2_control = Control(w_sol2)


# Convection difussion equation
k = Constant(1e-3)
cp_value = 5.0e5
cp = Constant(cp_value)
t1 = Constant(1.0)
t2 = Constant(10.0)

# Mesh-related functions
u1, p1 = split(w_sol1)
u2, p2 = split(w_sol2)

n = FacetNormal(mesh)
h = CellDiameter(mesh)
T = FunctionSpace(mesh, "CG", 1)
t, rho = Function(T), TestFunction(T)
n = FacetNormal(mesh)
beta = u1 + u2
F = (inner(beta, grad(t)) * rho + k * inner(grad(t), grad(rho))) * dx - inner(
    k * grad(t), n
) * rho * (ds(OUTLET1) + ds(OUTLET2))

R_U = dot(beta, grad(t)) - k * div(grad(t))
beta_gls = 0.9
h = CellSize(mesh)
tau_gls = beta_gls * (
    (4.0 * dot(beta, beta) / h ** 2) + 9.0 * (4.0 * k / h ** 2) ** 2
) ** (-0.5)
degree = 4

theta_U = dot(beta, grad(rho)) - k * div(grad(rho))
F_T = F + tau_gls * inner(R_U, theta_U) * dx(degree=degree)


bc1 = DirichletBC(T, t1, INLET1)
bc2 = DirichletBC(T, t2, INLET2)
bcs = [bc1, bc2]
problem_T = NonlinearVariationalProblem(F_T, t, bcs=bcs)
solver_parameters = {
    "ksp_type": "fgmres",
    "snes_atol": 1e-7,
    "pc_type": "hypre",
    "pc_hypre_type": "euclid",
    "ksp_max_it": 300,
}
solver_T = NonlinearVariationalSolver(problem_T, solver_parameters=solver_parameters)
solver_T.solve()
t.rename("Temperature")

File("temperature.pvd").write(t)

power_drop = 1e1
Power1 = assemble(p1 / power_drop * ds(INLET1))
Power2 = assemble(p2 / power_drop * ds(INLET2))
scale_factor = 4e-4
Jform = assemble(Constant(-scale_factor * cp_value) * inner(t * u1, n) * ds(OUTLET1))


phi_pvd = File("phi_evolution.pvd")


flow_pvd = File("flow_opti.pvd")


w_pvd_1 = Function(W)
w_pvd_2 = Function(W)


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


c = Control(s)

# Reduced Functionals
Jhat = LevelSetFunctional(Jform, c, phi, derivative_cb_pre=deriv_cb)
P1hat = LevelSetFunctional(Power1, c, phi)
P1control = Control(Power1)

P2hat = LevelSetFunctional(Power2, c, phi)
P2control = Control(Power2)

Jhat_v = Jhat(phi)
print("Initial cost function value {:.5f}".format(Jhat_v), flush=True)
print("Power drop 1 {:.5f}".format(Power1), flush=True)
print("Power drop 2 {:.5f}".format(Power2), flush=True)


velocity = Function(S)
beta_param = 0.08
reg_solver = RegularizationSolver(
    S, mesh, beta=beta_param, gamma=1e5, dx=dx, sim_domain=0
)


reinit_solver = ReinitSolverDG(mesh, n_steps=20, dt=1e-3)
hmin = 0.001
hj_solver = HJLocalDG(mesh, PHI, hmin=hmin)
tol = 1e-5
dt = 0.001
params = {
    "alphaC": 1.0,
    "debug": 5,
    "alphaJ": 1.0,
    "dt": dt,
    "K": 1e-3,
    "maxit": opts.n_iters,
    "maxtrials": 5,
    "itnormalisation": 10,
    "tol_merit": 5e-3,  # new merit can be within 0.5% of the previous merit
    # "normalize_tol" : -1,
    "tol": tol,
}

problem = InfDimProblem(
    Jhat,
    reg_solver,
    hj_solver,
    reinit_solver,
    ineqconstraints=[
        Constraint(P1hat, 1.0, P1control),
        Constraint(P2hat, 1.0, P2control),
    ],
    reinit_steps=5,
)
results = nlspace_solve_shape(problem, params)