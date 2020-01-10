from firedrake import *
from lestofire import RegularizationSolver


mesh = UnitSquareMesh(400, 400)
x = SpatialCoordinate(mesh)

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S,name="deform")
beta = 4.0
reg_solver = RegularizationSolver(S, mesh, beta=beta, gamma=0.0, dx=dx)

# Exact solution with free Neumann boundary conditions for this domain
u_exact = Function(S)
u_exact.interpolate(as_vector(( cos(x[0]*pi*2)*cos(x[1]*pi*2), cos(x[0]*pi*2)*cos(x[1]*pi*2))))
f = Function(S)
f.interpolate(as_vector(( (1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2), (1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2))))

theta = TestFunction(S)

f.interpolate(as_vector(( (1+beta*8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2), (1+beta*8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2))))
rhs_form = inner(f, theta)*dx

velocity = Function(S)
rhs = assemble(-rhs_form)
reg_solver.solve(velocity, rhs)
File("solution_vel.pvd").write(velocity)
error = norm(project(u_exact-velocity, S))
print("error: {0:.5f}".format(error))

