from lestofire.optimization.interface import InfDimProblem
from firedrake import *
from firedrake_adjoint import *

from lestofire import (
    LevelSetFunctional,
    RegularizationSolver,
    HJStabSolver,
    ReinitSolver,
    nlspace_solve_shape,
    Constraint,
    InfDimProblem,
)

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

from pyadjoint import no_annotations


output_dir = "2D/"

mesh = Mesh("./2D_mesh.msh")

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="deform")
mesh.coordinates.assign(mesh.coordinates + s)

x, y = SpatialCoordinate(mesh)
PHI = FunctionSpace(mesh, "CG", 1)
phi_expr = sin(y * pi / 0.2) * cos(x * pi / 0.2) - Constant(0.8)
phi = interpolate(phi_expr, PHI)
phi.rename("LevelSet")
File(output_dir + "phi_initial.pvd").write(phi)


# Parameters
mu = Constant(0.04)  # viscosity
alphamin = 1e-12
alphamax = 2.5 / (2e-4)
parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    # "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
stokes_parameters = parameters
temperature_parameters = parameters
epsilon = Constant(10000.0)
u_inflow = 2e-3
tin1 = Constant(10.0)
tin2 = Constant(100.0)

Re = u_inflow * width / mu.values()[0]
print("Reynolds number: {:.5f}".format(Re), flush=True)


P2 = VectorElement("CG", mesh.ufl_cell(), 2)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

U = TrialFunction(W)
u, p = split(U)
V = TestFunction(W)
v, q = split(V)


def hs(phi, epsilon):
    return Constant(alphamax) * Constant(1.0) / (
        Constant(1.0) + exp(-epsilon * phi)
    ) + Constant(alphamin)


def stokes(phi, BLOCK_INLET_MOUTH, BLOCK_OUTLET_MOUTH):
    a_fluid = mu * inner(grad(u), grad(v)) - div(v) * p - q * div(u)
    darcy_term = inner(u, v)
    return (
        a_fluid * dx
        + hs(phi, epsilon) * darcy_term * dx(0)
        + alphamax * darcy_term * (dx(BLOCK_INLET_MOUTH) + dx(BLOCK_OUTLET_MOUTH))
    )


# Dirichelt boundary conditions
inflow1 = as_vector(
    [
        u_inflow
        * sin(((y - (line_sep - (dist_center + inlet_width))) * pi) / inlet_width),
        0.0,
    ]
)
inflow2 = as_vector(
    [u_inflow * sin(((y - (line_sep + dist_center)) * pi) / inlet_width), 0.0]
)

noslip = Constant((0.0, 0.0))

# Stokes 1
bcs1_1 = DirichletBC(W.sub(0), noslip, WALLS)
bcs1_2 = DirichletBC(W.sub(0), inflow1, INLET1)
bcs1_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET1)
bcs1_4 = DirichletBC(W.sub(0), noslip, INLET2)
bcs1_5 = DirichletBC(W.sub(0), noslip, OUTLET2)
bcs1 = [bcs1_1, bcs1_2, bcs1_3, bcs1_4, bcs1_5]

# Stokes 2
bcs2_1 = DirichletBC(W.sub(0), noslip, WALLS)
bcs2_2 = DirichletBC(W.sub(0), inflow2, INLET2)
bcs2_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET2)
bcs2_4 = DirichletBC(W.sub(0), noslip, INLET1)
bcs2_5 = DirichletBC(W.sub(0), noslip, OUTLET1)
bcs2 = [bcs2_1, bcs2_2, bcs2_3, bcs2_4, bcs2_5]

# Forward problems
U1, U2 = Function(W), Function(W)
L = inner(Constant((0.0, 0.0, 0.0)), V) * dx
problem = LinearVariationalProblem(stokes(-phi, INMOUTH2, OUTMOUTH2), L, U1, bcs=bcs1)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
solver_stokes1 = LinearVariationalSolver(
    problem, solver_parameters=stokes_parameters
)  # , nullspace=nullspace)
solver_stokes1.solve()
problem = LinearVariationalProblem(stokes(phi, INMOUTH1, OUTMOUTH1), L, U2, bcs=bcs2)
solver_stokes2 = LinearVariationalSolver(problem, solver_parameters=stokes_parameters)
solver_stokes2.solve()

# Convection difussion equation
ks = Constant(1e0)
cp_value = 5.0e5
cp = Constant(cp_value)
T = FunctionSpace(mesh, "DG", 1)
t = Function(T, name="Temperature")
w = TestFunction(T)

# Mesh-related functions
n = FacetNormal(mesh)
h = CellDiameter(mesh)
u1, p1 = split(U1)
u2, p2 = split(U2)


def upwind(u):
    return (dot(u, n) + abs(dot(u, n))) / 2.0


u1n = upwind(u1)
u2n = upwind(u2)

# Penalty term
alpha = Constant(
    50000.0
)  # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
alpha = Constant(
    500.0
)  # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
# Bilinear form
a_int = dot(grad(w), ks * grad(t) - cp * (u1 + u2) * t) * dx

a_fac = (
    Constant(-1.0) * ks * dot(avg(grad(w)), jump(t, n)) * dS
    + Constant(-1.0) * ks * dot(jump(w, n), avg(grad(t))) * dS
    + ks("+") * (alpha("+") / avg(h)) * dot(jump(w, n), jump(t, n)) * dS
)

a_vel = (
    dot(
        jump(w),
        cp * (u1n("+") + u2n("+")) * t("+") - cp * (u1n("-") + u2n("-")) * t("-"),
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
eT = aT - LT_bnd

problem = NonlinearVariationalProblem(eT, t)
solver_temp = NonlinearVariationalSolver(
    problem, solver_parameters=temperature_parameters
)
solver_temp.solve()

power_drop = 1e-2
Power1 = assemble(p1 / power_drop * ds(INLET1))
Power2 = assemble(p2 / power_drop * ds(INLET2))
scale_factor = 4e-2
Jform = assemble(Constant(-scale_factor * cp_value) * inner(t * u1, n) * ds(OUTLET1))

U1control = Control(U1)
U2control = Control(U2)
u1_pvd = File(output_dir + "u1.pvd")
u2_pvd = File(output_dir + "u2.pvd")
tcontrol = Control(t)
tplot = Function(T)
t_pvd = File(output_dir + "t.pvd")


phi_pvd = File("phi_evolution.pvd")
def deriv_cb(phi):
    phi_pvd.write(phi[0])


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

# Jhat.optimize_tape()

velocity = Function(S)
beta_param = 1.0
reg_solver = RegularizationSolver(
    S, mesh, beta=beta_param, gamma=1e5, dx=dx, sim_domain=0, output_dir=None
)


reinit_solver = ReinitSolver(mesh, PHI, dt=1e-7, iterative=False)
hj_solver = HJStabSolver(mesh, PHI, c2_param=1.0, iterative=False)
# dt = 0.5*1e-1
dt = 10.0
tol = 1e-5



## Old parameters
# maxv = np.max(x.vector()[:])
# hmin = 0.00940  # Hard coded from FEniCS
# dt = 0.1 * 1.0 * hmin / maxv
## dt = 0.01
# self.newphi.assign(
#    hj_solver.solve(Constant(-1.0) * dx, x, steps=1, dt=dt), annotate=False
# )
# return self.newphi


params = {
    "alphaC": 1.0,
    "debug": 5,
    "alphaJ": 0.5,
    "dt": dt,
    "K": 1e-4,
    "maxit": 100,
    "maxtrials": 10,
    "itnormalisation": 500,
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
)
results = nlspace_solve_shape(problem, params)

