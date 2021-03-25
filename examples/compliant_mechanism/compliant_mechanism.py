from firedrake import *
from firedrake_adjoint import *

from lestofire import (
    LevelSetFunctional,
    RegularizationSolver,
    HJDG,
    HJLocalDG,
    ReinitSolverDG,
    nlspace_solve_shape,
    Constraint,
    InfDimProblem,
)

from pyadjoint import no_annotations
from params import SPRING, LOAD, DIRCH, FIXED
from params import height, width

output_dir = "compliant/"

# mesh = Mesh("./2D_mesh.msh")
# mesh = Mesh("./laurain_mechanism.msh")
mesh = Mesh("./laurain_mechanism_200.msh")
# mesh = RectangleMesh(120, 120, 1.0, 1.0, quadrilateral=True)

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="deform")
mesh.coordinates.assign(mesh.coordinates + s)

x, y = SpatialCoordinate(mesh)
PHI = FunctionSpace(mesh, "DG", 0)
lx = width
ly = height
phi_expr = (
    -cos(10.0 / lx * pi * x) * cos(12.0 * pi * y)
    - 0.3
    + max_value(100.0 * (x + y - lx - ly + 0.1), 0.0)
    + max_value(100.0 * (x - y - lx + 0.1), 0.0)
)
with stop_annotating():
    phi = interpolate(phi_expr, PHI)
    phi.rename("LevelSet")
    File(output_dir + "phi_initial.pvd").write(phi)


rho_min = 1e-6


def hs(phi):
    beta = Constant(10000.0)
    return Constant(1.0) / (Constant(1.0) + exp(-beta * phi)) + Constant(rho_min)


H1_elem = VectorElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, H1_elem)

u = TrialFunction(W)
v = TestFunction(W)

# Elasticity parameters
E, nu = 20.0, 0.3
mu, lmbda = Constant(E / (2 * (1 + nu))), Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))


def epsilon(u):
    return sym(nabla_grad(u))


def sigma(v):
    return 2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(2)


ks = 1e-2
n = FacetNormal(mesh)

a = (
    inner(hs(-phi) * sigma(u), nabla_grad(v)) * dx(0)
    + inner(sigma(u), nabla_grad(v)) * dx(FIXED)
    + ks * inner(v, inner(u, n) * n) * ds(SPRING)
)

t = Constant((1.0, 0.0))
L = inner(t, v) * ds(LOAD)

bc1 = DirichletBC(W, Constant((0.0, 0.0)), DIRCH)
parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "mat_type": "aij",
    "ksp_converged_reason": None,
    "pc_factor_mat_solver_type": "mumps",
}
u_sol = Function(W)
solve(a == L, u_sol, bcs=[bc1], solver_parameters=parameters)  # , nullspace=nullspace)
u_control = Control(u_sol)

F = Constant((1.0, 0.0))
scale_J = 1e0
Jform = scale_J * assemble(
    Constant(2.0) * inner(u_sol, F) * ds(LOAD) + inner(u_sol, F) * ds(SPRING)
)

with stop_annotating():
    total_volume = assemble(
        Constant(1.0) * dx(0, domain=mesh) + Constant(1.0) * dx(FIXED, domain=mesh)
    )
Vval = 1.0
VolPen = assemble(hs(-phi) * dx(0) + Constant(1.0) * dx(FIXED, domain=mesh)) / (
    total_volume / 3.0
)
VolControl = Control(VolPen)


velocity = Function(S)
bcs_vel = [DirichletBC(S, Constant((0.0, 0.0)), (1, 2, 3))]


phi_pvd = File("phi_evolution.pvd", target_continuity=H1)
geometry_pvd = File("geometry.pvd")
geometry = Function(PHI, name="geometry")
u_pvd = File("u_sol.pvd")
u_func = Function(W)


def deriv_cb(phi):
    with stop_annotating():
        phi_pvd.write(phi[0])
        geometry.assign(interpolate(hs(-phi[0]), PHI))
        geometry_pvd.write(geometry)
        u_func.assign(u_control.tape_value())
        u_pvd.write(u_func)


c = Control(s)
Jhat = LevelSetFunctional(Jform, c, phi, derivative_cb_pre=deriv_cb)
Vhat = LevelSetFunctional(VolPen, c, phi)
beta_param = 1e1
reg_solver = RegularizationSolver(
    S, mesh, beta=beta_param, gamma=1.0e8, dx=dx, bcs=bcs_vel, sim_domain=0
)
reinit_solver = ReinitSolverDG(mesh, n_steps=10, dt=1e-3)
hmin = 0.001
hj_solver = HJLocalDG(mesh, PHI, phi, hmin=hmin)
# dt = 0.5*1e-1
dt = 1.0e-1
tol = 1e-9

delta_x = Function(S)
delta_x_pvd = File("delta.pvd")
vol_constraint = Constraint(Vhat, Vval, VolControl)
problem = InfDimProblem(
    Jhat, reg_solver, hj_solver, reinit_solver, ineqconstraints=vol_constraint
)

parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "mat_type": "aij",
    "ksp_converged_reason": None,
    "pc_factor_mat_solver_type": "mumps",
}

params = {
    "alphaC": 0.1,
    "K": 0.1,
    "debug": 5,
    "alphaJ": 5.0,
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
