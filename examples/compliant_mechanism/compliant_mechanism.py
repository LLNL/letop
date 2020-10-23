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

from pyadjoint import no_annotations
from params import SPRING, LOAD, DIRCH, ROLL
from params import height, width

output_dir = "compliant/"

mesh = Mesh("./2D_mesh.msh")

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="deform")
mesh.coordinates.assign(mesh.coordinates + s)

x, y = SpatialCoordinate(mesh)
PHI = FunctionSpace(mesh, "CG", 1)
lx = width
ly = height
phi_expr = (
    -cos(10.0 / lx * pi * x) * cos(12.0 * pi * y)
    - 0.3
    + max_value(100.0 * (x + y - lx - ly + 0.1), 0.0)
    + max_value(100.0 * (x - y - lx + 0.1), 0.0)
)
phi = interpolate(phi_expr, PHI)
phi.rename("LevelSet")
File(output_dir + "phi_initial.pvd").write(phi)


rho_min = 1e-3
beta = Constant(200.0)


def hs(phi, beta):
    return Constant(1.0) / (Constant(1.0) + exp(-beta * phi)) + Constant(rho_min)


H1 = VectorElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, H1)

u = TrialFunction(W)
v = TestFunction(W)

# Elasticity parameters
E, nu = 20.0, 0.3
mu, lmbda = Constant(E / (2 * (1 + nu))), Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))


def epsilon(u):
    return sym(nabla_grad(u))


def sigma(v):
    return 2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(2)


ks = 0.01
n = FacetNormal(mesh)

f = Constant((0.0, 0.0))
a = (
    inner(hs(-phi, beta) * sigma(u), nabla_grad(v)) * dx(0)
    + inner(sigma(u), nabla_grad(v)) * (dx(1) + dx(2))
    + ks * inner(v, inner(u, n) * n) * ds(SPRING)
)

t = Constant((5.0, 0.0))
L = inner(t, v) * ds(LOAD)

bc1 = DirichletBC(W, Constant((0.0, 0.0)), 1)
# bc2 = DirichletBC(W.sub(1), Constant(0.0), 4)
parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "mat_type": "aij",
    "ksp_converged_reason": None,
    # "pc_factor_mat_solver_type": "mumps"
}
u_sol = Function(W)
solve(a == L, u_sol, bcs=[bc1], solver_parameters=parameters)  # , nullspace=nullspace)
File("u_sol.pvd").write(u_sol)
Sigma = TensorFunctionSpace(mesh, "CG", 1)
File("sigma.pvd").write(project(sigma(u_sol), Sigma))

F = Constant((1.0, 0.0))
Jform = assemble(2.0 * inner(u_sol, F) * ds(LOAD) + inner(u_sol, F) * ds(SPRING))
# Jform = assemble(inner(u_sol, n)*ds(SPRING))
VolPen = assemble(hs(-phi, beta) * dx(0) + Constant(1.0) * dx(SPRING, domain=mesh))
Lag = 0.01
J = Jform + Lag * VolPen
print(f"AL value of cost function is {Jform + Lag*VolPen}, J: {Jform}, V: {VolPen}")
VolControl = Control(VolPen)

Vval = 1.0


velocity = Function(S)
bcs_vel_1 = DirichletBC(S, Constant((0.0, 0.0)), 1)
bcs_vel_2 = DirichletBC(S, Constant((0.0, 0.0)), 2)
bcs_vel = [bcs_vel_2]


phi_pvd = File("phi_evolution.pvd")


def deriv_cb(phi):
    phi_pvd.write(phi[0])


c = Control(s)
Jhat = LevelSetFunctional(J, c, phi, derivative_cb_pre=deriv_cb)
# Vhat = LevelSetFunctional(VolPen, c, phi)
beta_param = 1e0
reg_solver = RegularizationSolver(
    S, mesh, beta=beta_param, gamma=1.0e5, dx=dx, bcs=bcs_vel, sim_domain=0
)
reinit_solver = ReinitSolver(mesh, PHI, dt=1e-7, iterative=False)
hj_solver = HJStabSolver(mesh, PHI, c2_param=0.01, iterative=False)
# dt = 0.5*1e-1
dt = 1.0e-2
tol = 1e-5

# vol_constraint = Constraint(Vhat, Vval, VolControl)
problem = InfDimProblem(
    # Jhat, reg_solver, hj_solver, reinit_solver, ineqconstraints=vol_constraint
    Jhat,
    reg_solver,
    hj_solver,
    reinit_solver,
)

parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "mat_type": "aij",
    "ksp_converged_reason": None,
    # "pc_factor_mat_solver_type": "mumps",
}

params = {
    "alphaC": 1.0,
    "K": 0.1,
    "debug": 5,
    "alphaJ": 1.0,
    "dt": dt,
    "maxtrials": 10,
    "maxit": 200,
    "itnormalisation": 10,
    "normalize_tol": 1e-2,
    "tol": tol,
}
delta_pvd = File("delta_x.pvd")
delta_x = Function(S)
results = nlspace_solve_shape(
    problem, params, delta_x_pvd=delta_pvd, delta_x_func=delta_x
)

