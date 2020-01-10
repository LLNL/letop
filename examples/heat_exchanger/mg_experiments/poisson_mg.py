from firedrake import *

mesh = UnitSquareMesh(8, 8)
hierarchy = MeshHierarchy(mesh, 1)
mesh = hierarchy[-1]

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

a = dot(grad(u), grad(v))*dx

bcs = DirichletBC(V, zero(), (1, 2, 3, 4))

x, y = SpatialCoordinate(mesh)

f = Constant(0.1)

L = f*v*dx

u = Function(V)

parameters = {
   "ksp_type": "cg",
   "ksp_converged_reason": None,
   "pc_type": "mg",
}

A = assemble(a, bcs=bcs)
b = assemble(L, bcs=bcs)
solve(a == L, u, bcs=bcs, solver_parameters=parameters)
solve(a == L, u, bcs=bcs, solver_parameters=parameters)
#solve(A, u.vector(), b, solver_parameters=parameters)
