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

output_dir = "cantilever/"

mesh = Mesh("./mesh_cantilever.msh")

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="deform")
mesh.coordinates.assign(mesh.coordinates + s)

x, y = SpatialCoordinate(mesh)
PHI = FunctionSpace(mesh, "CG", 1)
lx = 2.0
ly = 1.0
phi_expr = (
    -cos(6.0 / lx * pi * x) * cos(4.0 * pi * y)
    - 0.6
    + max_value(200.0 * (0.01 - x ** 2 - (y - ly / 2) ** 2), 0.0)
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
E, nu = 1.0, 0.3
mu, lmbda = Constant(E / (2 * (1 + nu))), Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))


def epsilon(u):
    return sym(nabla_grad(u))


def sigma(v):
    return 2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(2)


f = Constant((0.0, 0.0))
a = inner(hs(-phi, beta) * sigma(u), nabla_grad(v)) * dx
t = Constant((0.0, -75.0))
L = inner(t, v) * ds(2)

bc = DirichletBC(W, Constant((0.0, 0.0)), 1)
parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "mat_type": "aij",
    "ksp_converged_reason": None
    # "pc_factor_mat_solver_type": "mumps"
}
u_sol = Function(W)
solve(a == L, u_sol, bcs=[bc], solver_parameters=parameters)  # , nullspace=nullspace)
File("u_sol.pvd").write(u_sol)
Sigma = TensorFunctionSpace(mesh, "CG", 1)
File("sigma.pvd").write(project(sigma(u_sol), Sigma))

Jform = assemble(inner(hs(-phi, beta) * sigma(u_sol), epsilon(u_sol)) * dx)
VolPen = assemble(hs(-phi, beta) * dx)
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
Jhat = LevelSetFunctional(Jform, c, phi, derivative_cb_pre=deriv_cb)
Vhat = LevelSetFunctional(VolPen, c, phi)
beta_param = 1e2
reg_solver = RegularizationSolver(
    S, mesh, beta=beta_param, gamma=1.0e5, dx=dx, bcs=bcs_vel, output_dir=None
)
reinit_solver = ReinitSolver(mesh, PHI, dt=1e-7, iterative=False)
hj_solver = HJStabSolver(mesh, PHI, c2_param=2.0, iterative=False)
# dt = 0.5*1e-1
dt = 20.0
tol = 1e-5

vol_constraint = Constraint(Vhat, Vval, VolControl)
problem = InfDimProblem(
    Jhat, reg_solver, hj_solver, reinit_solver, ineqconstraints=vol_constraint
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
    "itnormalisation": 50,
    "tol": tol,
}
results = nlspace_solve_shape(problem, params)
# results = nlspace_solve_shape(InfDimProblem(phi, Jhat, c), params)

