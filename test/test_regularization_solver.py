from firedrake import (UnitSquareMesh, SpatialCoordinate, VectorFunctionSpace,
                        Function, TestFunction, assemble, norm, project, inner,
                        as_vector, cos, pi, sin, dx, File, Mesh, conditional)
import numpy as np
from lestofire import RegularizationSolver


def regularization_form(r):
    mesh = UnitSquareMesh(2 ** r, 2 ** r)
    x = SpatialCoordinate(mesh)

    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    beta = 4.0
    reg_solver = RegularizationSolver(S, mesh, beta=beta, gamma=0.0, dx=dx)

    # Exact solution with free Neumann boundary conditions for this domain
    u_exact = Function(S)
    u_exact_component = cos(x[0]*pi*2)*cos(x[1]*pi*2)
    u_exact.interpolate(as_vector(( u_exact_component, u_exact_component)))
    f = Function(S)
    theta = TestFunction(S)
    f_component = (1+beta*8*pi*pi)*u_exact_component
    f.interpolate(as_vector(( f_component, f_component)))
    rhs_form = inner(f, theta)*dx

    velocity = Function(S)
    rhs = assemble(-rhs_form)
    reg_solver.solve(velocity, rhs)
    File("solution_vel_unitsquare.pvd").write(velocity)
    return norm(project(u_exact-velocity, S))

def test_regularization_convergence():
    diff = np.array([regularization_form(i) for i in range(3, 6)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > 1.8).all()

def domainify(vector, x):
    dim = len(x)
    return conditional(x[0] > 0.0, vector, as_vector(tuple(0.0 for _ in range(dim))))

def test_2D_dirichlet_regions():
    # Can't do mesh refinement because MeshHierarchy ruins the domain tags
    mesh = Mesh("./2D_mesh.msh")
    dim = mesh.geometric_dimension()
    x = SpatialCoordinate(mesh)

    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    beta = 1.0
    reg_solver = RegularizationSolver(S, mesh, beta=beta, gamma=0.0, dx=dx, sim_domain=0)

    # Exact solution with free Neumann boundary conditions for this domain
    u_exact = Function(S)
    u_exact_component = (-cos(x[0]*pi*2) + 1)*(-cos(x[1]*pi * 2) + 1)


    u_exact.interpolate(as_vector(tuple(u_exact_component for _ in range(dim))))
    f = Function(S)
    f_component = -beta*(4.0 * pi * pi * cos(2 *pi *x[0]) * (-cos(2 *pi * x[1]) + 1) +
                    4.0 * pi * pi * cos(2 *pi *x[1]) * (-cos(2 *pi * x[0]) + 1)) + \
                    u_exact_component
    f.interpolate(as_vector(tuple(f_component for _ in range(dim))))

    theta = TestFunction(S)
    rhs_form = inner(f, theta)*dx

    velocity = Function(S)
    rhs = assemble(-rhs_form)
    reg_solver.solve(velocity, rhs)
    assert(norm(project(domainify(u_exact, x)-velocity, S)) < 1e-3)


def test_3D_dirichlet_regions():
    # Can't do mesh refinement because MeshHierarchy ruins the domain tags
    mesh = Mesh("./3D_mesh.msh")
    dim = mesh.geometric_dimension()
    x = SpatialCoordinate(mesh)

    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    beta = 1.0
    reg_solver = RegularizationSolver(S, mesh, beta=beta, gamma=0.0, dx=dx, sim_domain=0, iterative=True)

    # Exact solution with free Neumann boundary conditions for this domain
    u_exact = Function(S)
    u_exact_component = (-cos(x[0]*pi*2) + 1)*\
                        (-cos(x[1]*pi * 2) + 1)*\
                        (-cos(x[2]*pi*2) + 1)

    u_exact.interpolate(as_vector(tuple(u_exact_component for _ in range(dim))))
    f = Function(S)
    f_component = -beta*(4.0 * pi * pi * cos(2 *pi *x[0]) * (-cos(2 *pi * x[1]) + 1) * (-cos(2 *pi * x[2]) + 1) +
                    4.0 * pi * pi * cos(2 *pi *x[1]) * (-cos(2 *pi * x[0]) + 1) * (-cos(2 *pi * x[2]) + 1) +
                    4.0 * pi * pi * cos(2 *pi *x[2]) * (-cos(2 *pi * x[1]) + 1) * (-cos(2 *pi * x[0]) + 1)) + \
                    u_exact_component
    f.interpolate(as_vector(tuple(f_component for _ in range(dim))))

    theta = TestFunction(S)
    rhs_form = inner(f, theta)*dx

    velocity = Function(S)
    rhs = assemble(-rhs_form)
    reg_solver.solve(velocity, rhs)
    error = norm(project(domainify(u_exact, x)-velocity, S))
    assert(error < 5e-2)

if __name__ == "__main__":
    test_regularization_convergence()
    test_2D_dirichlet_regions()
    test_3D_dirichlet_regions()
