from firedrake import *
from firedrake_adjoint import *
from lestofire import (
    NavierStokesBrinkmannSolver,
    NavierStokesBrinkmannForm,
)

mesh = Mesh("./mesh_stokes.msh")
S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="deform")
mesh.coordinates.assign(mesh.coordinates + s)

x, y = SpatialCoordinate(mesh)
PHI = FunctionSpace(mesh, "DG", 1)
lx = 2.0
ly = 1.0
phi_expr = -cos(6.0 / lx * pi * x + 1.0) * cos(4.0 * pi * y) - 0.6

with stop_annotating():
    phi = interpolate(phi_expr, PHI)
phi.rename("LevelSet")

nu = Constant(0.1)
V = VectorFunctionSpace(mesh, "CG", 1)
P = FunctionSpace(mesh, "CG", 1)
W = V * P
w_sol = Function(W)
F = NavierStokesBrinkmannForm(W, w_sol, phi, nu, brinkmann_penalty=1e3)

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
solver_parameters = {"ksp_max_it": 50}
solver = NavierStokesBrinkmannSolver(problem, solver_parameters=solver_parameters)
solver.solve()
pvd_file = File("ns_solution.pvd")
u, p = w_sol.split()
pvd_file.write(u, p)

# u, p = split(w)
#
# Vol = assemble(hs(-phi, epsilon) * Constant(1.0) * dx(domain=mesh))
# VControl = Control(Vol)
# Vval = assemble(Constant(0.3) * dx(domain=mesh), annotate=False)
# with stop_annotating():
#    print("Initial constraint function value {}".format(Vol))
#
# J = assemble(
#    Constant(alphamax) * hs(phi, epsilon) * inner(u, u) * dx
#    + mu / Constant(2.0) * inner(grad(u), grad(u)) * dx
# )
#
# c = Control(s)
#
# phi_pvd = File("phi_evolution.pvd")
#
#
# def deriv_cb(phi):
#    with stop_annotating():
#        phi_pvd.write(phi[0])
#
#
# Jhat = LevelSetFunctional(J, c, phi, derivative_cb_pre=deriv_cb)
# Vhat = LevelSetFunctional(Vol, c, phi)
#
# velocity = Function(S)
# bcs_vel_1 = DirichletBC(S, noslip, 1)
# bcs_vel_2 = DirichletBC(S, noslip, 2)
# bcs_vel_3 = DirichletBC(S, noslip, 3)
# bcs_vel_4 = DirichletBC(S, noslip, 4)
# bcs_vel_5 = DirichletBC(S, noslip, 5)
# bcs_vel = [bcs_vel_1, bcs_vel_2, bcs_vel_3, bcs_vel_4, bcs_vel_5]
# reg_solver = RegularizationSolver(
#    S, mesh, beta=1, gamma=0.0, dx=dx, bcs=bcs_vel, output_dir=None
# )
