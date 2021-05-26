import firedrake as fd
from firedrake import sqrt, inner, dx
from lestofire.optimization import ReinitializationSolver
from lestofire.physics import (
    NavierStokesBrinkmannForm,
    NavierStokesBrinkmannSolver,
    mark_no_flow_regions,
    InteriorBC,
)
import pytest


def test_solver_no_flow_region():
    mesh = fd.Mesh("../2D_mesh.msh")
    no_flow = [2]
    no_flow_markers = [1]
    mesh = mark_no_flow_regions(mesh, no_flow, no_flow_markers)
    P2 = fd.VectorElement("CG", mesh.ufl_cell(), 1)
    P1 = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = fd.FunctionSpace(mesh, TH)
    (v, q) = fd.TestFunctions(W)

    # Stokes 1
    w_sol1 = fd.Function(W)
    nu = fd.Constant(0.05)
    F = NavierStokesBrinkmannForm(W, w_sol1, nu, beta_gls=2.0)

    from firedrake import sin, grad, pi, sym, div, inner

    x, y = fd.SpatialCoordinate(mesh)
    u_mms = fd.as_vector(
        [sin(2.0 * pi * x) * sin(pi * y), sin(pi * x) * sin(2.0 * pi * y)]
    )
    p_mms = -0.5 * (u_mms[0] ** 2 + u_mms[1] ** 2)
    f_mms_u = grad(u_mms) * u_mms + grad(p_mms) - 2.0 * nu * div(sym(grad(u_mms)))
    f_mms_p = div(u_mms)
    F += -inner(f_mms_u, v) * dx - f_mms_p * q * dx
    bc1 = fd.DirichletBC(W.sub(0), u_mms, "on_boundary")
    bc2 = fd.DirichletBC(W.sub(1), p_mms, "on_boundary")
    bc_no_flow = InteriorBC(W.sub(0), fd.Constant((0.0, 0.0)), no_flow_markers)

    solver_parameters = {"ksp_max_it": 500, "ksp_monitor": None}

    problem1 = fd.NonlinearVariationalProblem(F, w_sol1, bcs=[bc1, bc2, bc_no_flow])
    solver1 = NavierStokesBrinkmannSolver(
        problem1, options_prefix="navier_stokes", solver_parameters=solver_parameters
    )
    solver1.solve()
    u_sol, _ = w_sol1.split()
    fd.File("test_u_sol.pvd").write(u_sol)
    u_mms_func = fd.interpolate(u_mms, W.sub(0))
    error = fd.errornorm(u_sol, u_mms_func)
    print(f"Error: {error}")
    assert error < 0.07


def run_solver(r):
    mesh = fd.UnitSquareMesh(2 ** r, 2 ** r)
    P2 = fd.VectorElement("CG", mesh.ufl_cell(), 1)
    P1 = fd.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = fd.FunctionSpace(mesh, TH)
    (v, q) = fd.TestFunctions(W)

    # Stokes 1
    w_sol1 = fd.Function(W)
    nu = fd.Constant(0.05)
    F = NavierStokesBrinkmannForm(W, w_sol1, nu, beta_gls=2.0)

    from firedrake import sin, grad, pi, sym, div, inner

    x, y = fd.SpatialCoordinate(mesh)
    u_mms = fd.as_vector(
        [sin(2.0 * pi * x) * sin(pi * y), sin(pi * x) * sin(2.0 * pi * y)]
    )
    p_mms = -0.5 * (u_mms[0] ** 2 + u_mms[1] ** 2)
    f_mms_u = grad(u_mms) * u_mms + grad(p_mms) - 2.0 * nu * div(sym(grad(u_mms)))
    f_mms_p = div(u_mms)
    F += -inner(f_mms_u, v) * dx - f_mms_p * q * dx
    bc1 = fd.DirichletBC(W.sub(0), u_mms, "on_boundary")
    bc2 = fd.DirichletBC(W.sub(1), p_mms, "on_boundary")

    solver_parameters = {"ksp_max_it": 200}

    problem1 = fd.NonlinearVariationalProblem(F, w_sol1, bcs=[bc1, bc2])
    solver1 = NavierStokesBrinkmannSolver(
        problem1, options_prefix="navier_stokes", solver_parameters=solver_parameters
    )
    solver1.solve()
    u_sol, _ = w_sol1.split()
    fd.File("test_u_sol.pvd").write(u_sol)
    u_mms_func = fd.interpolate(u_mms, W.sub(0))
    error = fd.errornorm(u_sol, u_mms_func)
    print(f"Error: {error}")
    return error


def run_convergence_test():
    import numpy as np

    diff = np.array([run_solver(i) for i in range(4, 7)])
    return np.log2(diff[:-1] / diff[1:])


def test_l2_conv():
    assert (run_convergence_test() > 0.8).all()
