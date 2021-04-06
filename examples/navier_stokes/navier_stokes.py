from firedrake import *
from firedrake_adjoint import *
from lestofire import (
    NavierStokesBrinkmannSolver,
    NavierStokesBrinkmannForm,
    hs,
    LevelSetFunctional,
    ReinitSolverDG,
    RegularizationSolver,
    HJLocalDG,
)

from nullspace_optimizer.lestofire import InfDimProblem, Constraint, nlspace_solve_shape

mesh = Mesh("./mesh_stokes.msh")
S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="deform")
mesh.coordinates.assign(mesh.coordinates + s)

x, y = SpatialCoordinate(mesh)
PHI = FunctionSpace(mesh, "DG", 0)
lx = 1.0
ly = 1.0
# phi_expr = -0.5 * cos(3.0 / lx * pi * x + 1.0) * cos(3.0 * pi * y) - 0.3
lx = 2.0
ly = 1.0
phi_expr = -cos(6.0 / lx * pi * x + 1.0) * cos(4.0 * pi * y) - 0.6

with stop_annotating():
    phi = interpolate(phi_expr, PHI)
phi.rename("LevelSet")

nu = Constant(1.0)
V = VectorFunctionSpace(mesh, "CG", 1)
P = FunctionSpace(mesh, "CG", 1)
W = V * P
w_sol = Function(W)
brinkmann_penalty = 1e6
F = NavierStokesBrinkmannForm(W, w_sol, phi, nu, brinkmann_penalty=brinkmann_penalty)

x, y = SpatialCoordinate(mesh)
u_inflow = 1.0
y_inlet_1_1 = 0.2
y_inlet_1_2 = 0.4
inflow1 = as_vector(
    [
        u_inflow * 100 * (y - y_inlet_1_1) * (y - y_inlet_1_2),
        0.0,
    ]
)
y_inlet_2_1 = 0.6
y_inlet_2_2 = 0.8
inflow2 = as_vector(
    [
        u_inflow * 100 * (y - y_inlet_2_1) * (y - y_inlet_2_2),
        0.0,
    ]
)

noslip = Constant((0.0, 0.0))
bc1 = DirichletBC(W.sub(0), noslip, 5)
bc2 = DirichletBC(W.sub(0), inflow1, (1))
bc3 = DirichletBC(W.sub(0), inflow2, (2))
bcs = [bc1, bc2, bc3]

problem = NonlinearVariationalProblem(F, w_sol, bcs=bcs)
solver_parameters = {
    "ksp_max_it": 100,
    "snes_rtol": 1e-3,
    "ksp_norm_type": None,
    "ksp_convergence_test": "skip",
    "ksp_atol": 1e-3,
    "ksp_rtol": 1e-3,
}
# solver_parameters = {
#    "ksp_type": "preonly",
#    "pc_type": "lu",
#    "mat_type": "aij",
#    "ksp_converged_reason": None,
#    "pc_factor_mat_solver_type": "mumps",
# }
solver_parameters = {
    "ksp_type": "fgmres",
    "pc_type": "hypre",
    "pc_hypre_type": "euclid",
    "pc_hypre_euclid_level": 5,
    "mat_type": "aij",
    "ksp_converged_reason": None,
    "ksp_atol": 1e-3,
    "ksp_rtol": 1e-3,
    "snes_atol": 1e-3,
    "snes_rtol": 1e-3,
}
solver = NavierStokesBrinkmannSolver(problem, solver_parameters=solver_parameters)
solver.solve()
pvd_file = File("ns_solution.pvd")
u, p = w_sol.split()
pvd_file.write(u, p)

u, p = split(w_sol)

Vol = assemble(hs(-phi) * Constant(1.0) * dx(domain=mesh))
VControl = Control(Vol)
Vval = assemble(Constant(0.3) * dx(domain=mesh), annotate=False)
with stop_annotating():
    print("Initial constraint function value {}".format(Vol))

J = assemble(
    Constant(brinkmann_penalty) * hs(phi) * inner(u, u) * dx
    + nu / Constant(2.0) * inner(grad(u), grad(u)) * dx
)

c = Control(s)

phi_pvd = File("phi_evolution_euclid.pvd", target_continuity=H1)


def deriv_cb(phi):
    with stop_annotating():
        phi_pvd.write(phi[0])


Jhat = LevelSetFunctional(J, c, phi, derivative_cb_pre=deriv_cb)
Vhat = LevelSetFunctional(Vol, c, phi)

velocity = Function(S)
bcs_vel_1 = DirichletBC(S, noslip, (1, 2, 3, 4))
bcs_vel = [bcs_vel_1]
reg_solver = RegularizationSolver(S, mesh, beta=0.5, gamma=1e5, dx=dx, bcs=bcs_vel)


reinit_pvd = File("reinit.pvd")
reinit_solver = ReinitSolverDG(mesh, n_steps=20, dt=2e-4, phi_pvd=reinit_pvd)
reinit_solver.solve(phi)
hmin = 0.001
hj_solver = HJLocalDG(mesh, PHI, hmin=hmin, n_steps=1)
tol = 1e-5
dt = 0.001
params = {
    "alphaC": 1.0,
    "debug": 5,
    "alphaJ": 1.0,
    "dt": dt,
    "K": 0.1,
    "maxit": 2000,
    "maxtrials": 5,
    "itnormalisation": 10,
    "tol_merit": 1e-4,  # new merit can be within 5% of the previous merit
    # "normalize_tol" : -1,
    "tol": tol,
}

problem = InfDimProblem(
    Jhat,
    reg_solver,
    hj_solver,
    reinit_solver,
    ineqconstraints=[Constraint(Vhat, Vval, VControl)],
)
results = nlspace_solve_shape(problem, params)