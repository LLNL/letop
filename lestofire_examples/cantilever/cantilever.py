import firedrake as fd
import firedrake_adjoint as fda
from firedrake import (
    inner,
    dx,
    ds,
    cos,
    pi,
    exp,
    max_value,
    sym,
    nabla_grad,
    tr,
    Identity,
)

from lestofire.levelset import LevelSetFunctional, RegularizationSolver
from lestofire.optimization import InfDimProblem, Constraint
from lestofire.optimization import nlspace_solve

import os


def compliance_optimization(n_iters=200):

    output_dir = "cantilever/"

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    m = fd.Mesh(f"{dir_path}/mesh_cantilever.msh")
    mesh = fd.MeshHierarchy(m, 0)[-1]

    # Perturb the mesh coordinates. Necessary to calculate shape derivatives
    S = fd.VectorFunctionSpace(mesh, "CG", 1)
    s = fd.Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    # Initial level set function
    x, y = fd.SpatialCoordinate(mesh)
    PHI = fd.FunctionSpace(mesh, "CG", 1)
    lx = 2.0
    ly = 1.0
    phi_expr = (
        -cos(6.0 / lx * pi * x) * cos(4.0 * pi * y)
        - 0.6
        + max_value(200.0 * (0.01 - x ** 2 - (y - ly / 2) ** 2), 0.0)
        + max_value(100.0 * (x + y - lx - ly + 0.1), 0.0)
        + max_value(100.0 * (x - y - lx + 0.1), 0.0)
    )
    # Avoid recording the operation interpolate into the tape.
    # Otherwise, the shape derivatives will not be correct
    with fda.stop_annotating():
        phi = fd.interpolate(phi_expr, PHI)
        phi.rename("LevelSet")
        fd.File(output_dir + "phi_initial.pvd").write(phi)

    # Physics. Elasticity
    rho_min = 1e-5
    beta = fd.Constant(200.0)

    def hs(phi, beta):
        return fd.Constant(1.0) / (
            fd.Constant(1.0) + exp(-beta * phi)
        ) + fd.Constant(rho_min)

    H1_elem = fd.VectorElement("CG", mesh.ufl_cell(), 1)
    W = fd.FunctionSpace(mesh, H1_elem)

    u = fd.TrialFunction(W)
    v = fd.TestFunction(W)

    # Elasticity parameters
    E, nu = 1.0, 0.3
    mu, lmbda = fd.Constant(E / (2 * (1 + nu))), fd.Constant(
        E * nu / ((1 + nu) * (1 - 2 * nu))
    )

    def epsilon(u):
        return sym(nabla_grad(u))

    def sigma(v):
        return 2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(2)

    a = inner(hs(-phi, beta) * sigma(u), nabla_grad(v)) * dx
    t = fd.Constant((0.0, -75.0))
    L = inner(t, v) * ds(2)

    bc = fd.DirichletBC(W, fd.Constant((0.0, 0.0)), 1)
    parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "mat_type": "aij",
        "ksp_converged_reason": None,
        "pc_factor_mat_solver_type": "mumps",
    }
    u_sol = fd.Function(W)
    F = fd.action(a, u_sol) - L
    problem = fd.NonlinearVariationalProblem(F, u_sol, bcs=bc)
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=parameters
    )
    solver.solve()
    # fd.solve(
    #    a == L, u_sol, bcs=[bc], solver_parameters=parameters
    # )  # , nullspace=nullspace)
    with fda.stop_annotating():
        fd.File("u_sol.pvd").write(u_sol)

    # Cost function: Compliance
    J = fd.assemble(
        fd.Constant(1e-2)
        * inner(hs(-phi, beta) * sigma(u_sol), epsilon(u_sol))
        * dx
    )

    # Constraint: Volume
    with fda.stop_annotating():
        total_volume = fd.assemble(fd.Constant(1.0) * dx(domain=mesh))
    VolPen = fd.assemble(hs(-phi, beta) * dx)
    # Needed to track the value of the volume
    VolControl = fda.Control(VolPen)
    Vval = total_volume / 2.0

    phi_pvd = fd.File("phi_evolution.pvd", target_continuity=fd.H1)

    def deriv_cb(phi):
        with fda.stop_annotating():
            phi_pvd.write(phi[0])

    c = fda.Control(s)
    # Lestofire routines
    Jhat = LevelSetFunctional(J, c, phi, derivative_cb_pre=deriv_cb)
    Vhat = LevelSetFunctional(VolPen, c, phi)
    beta_param = 0.1
    # Boundary conditions for the shape derivatives.
    # They must be zero at the boundary conditions.
    bcs_vel = fd.DirichletBC(S, fd.Constant((0.0, 0.0)), (1, 2))
    # Regularize the shape derivatives
    reg_solver = RegularizationSolver(
        S,
        mesh,
        beta=beta_param,
        gamma=1.0e5,
        dx=dx,
        bcs=bcs_vel,
        output_dir=None,
    )
    # Hamilton-Jacobi equation to advect the level set
    dt = 0.05
    tol = 1e-5

    # Optimization problem
    vol_constraint = Constraint(Vhat, Vval, VolControl)
    problem = InfDimProblem(Jhat, reg_solver, ineqconstraints=vol_constraint)

    parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "mat_type": "aij",
        "ksp_converged_reason": None,
        "pc_factor_mat_solver_type": "mumps",
    }

    params = {
        "alphaC": 3.0,
        "K": 0.1,
        "debug": 5,
        "alphaJ": 1.0,
        "dt": dt,
        "maxtrials": 10,
        "maxit": n_iters,
        "itnormalisation": 50,
        "tol": tol,
    }
    results = nlspace_solve(problem, params)

    return results


if __name__ == "__main__":
    compliance_optimization()
