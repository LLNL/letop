import firedrake as fd
from firedrake import cos, pi, dx, inner, grad
import firedrake_adjoint as fda
from pyadjoint import stop_annotating
from lestofire.physics import (
    NavierStokesBrinkmannForm,
    NavierStokesBrinkmannSolver,
    hs,
)
from lestofire.levelset import LevelSetFunctional
from lestofire.levelset import RegularizationSolver
from lestofire.optimization import InfDimProblem, Constraint
from nullspace_optimizer.lestofire import nlspace_solve_shape


def main():
    mesh = fd.Mesh("./mesh_stokes.msh")
    mh = fd.MeshHierarchy(mesh, 1)
    mesh = mh[-1]
    # mesh = fd.Mesh("./mesh_stokes_inlets.msh")
    S = fd.VectorFunctionSpace(mesh, "CG", 1)
    s = fd.Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    x, y = fd.SpatialCoordinate(mesh)
    PHI = fd.FunctionSpace(mesh, "DG", 1)
    lx = 1.0
    # phi_expr = -0.5 * cos(3.0 / lx * pi * x + 1.0) * cos(3.0 * pi * y) - 0.3
    lx = 2.0
    phi_expr = -cos(6.0 / lx * pi * x + 1.0) * cos(4.0 * pi * y) - 0.6

    with stop_annotating():
        phi = fd.interpolate(phi_expr, PHI)
    phi.rename("LevelSet")

    nu = fd.Constant(1.0)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    P = fd.FunctionSpace(mesh, "CG", 1)
    W = V * P
    w_sol = fd.Function(W)
    brinkmann_penalty = 1e6
    F = NavierStokesBrinkmannForm(
        W, w_sol, phi, nu, brinkmann_penalty=brinkmann_penalty
    )

    x, y = fd.SpatialCoordinate(mesh)
    u_inflow = 1.0
    y_inlet_1_1 = 0.2
    y_inlet_1_2 = 0.4
    inflow1 = fd.as_vector(
        [
            u_inflow * 100 * (y - y_inlet_1_1) * (y - y_inlet_1_2),
            0.0,
        ]
    )
    y_inlet_2_1 = 0.6
    y_inlet_2_2 = 0.8
    inflow2 = fd.as_vector(
        [
            u_inflow * 100 * (y - y_inlet_2_1) * (y - y_inlet_2_2),
            0.0,
        ]
    )

    noslip = fd.Constant((0.0, 0.0))
    bc1 = fd.DirichletBC(W.sub(0), noslip, 5)
    bc2 = fd.DirichletBC(W.sub(0), inflow1, (1))
    bc3 = fd.DirichletBC(W.sub(0), inflow2, (2))
    bcs = [bc1, bc2, bc3]

    problem = fd.NonlinearVariationalProblem(F, w_sol, bcs=bcs)
    solver_parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "mat_type": "aij",
        "ksp_converged_reason": None,
        "pc_factor_mat_solver_type": "mumps",
    }
    # solver_parameters = {
    #    "ksp_type": "fgmres",
    #    "pc_type": "hypre",
    #    "pc_hypre_type": "euclid",
    #    "pc_hypre_euclid_level": 5,
    #    "mat_type": "aij",
    #    "ksp_converged_reason": None,
    #    "ksp_atol": 1e-3,
    #    "ksp_rtol": 1e-3,
    #    "snes_atol": 1e-3,
    #    "snes_rtol": 1e-3,
    # }
    solver = NavierStokesBrinkmannSolver(
        problem, solver_parameters=solver_parameters
    )
    solver.solve()
    pvd_file = fd.File("ns_solution.pvd")
    u, p = w_sol.split()
    pvd_file.write(u, p)

    u, p = fd.split(w_sol)

    Vol = fd.assemble(hs(-phi) * fd.Constant(1.0) * dx(0, domain=mesh))
    VControl = fda.Control(Vol)
    Vval = fd.assemble(fd.Constant(0.5) * dx(domain=mesh), annotate=False)
    with stop_annotating():
        print("Initial constraint function value {}".format(Vol))

    J = fd.assemble(
        fd.Constant(brinkmann_penalty) * hs(phi) * inner(u, u) * dx(0)
        + nu / fd.Constant(2.0) * inner(grad(u), grad(u)) * dx
    )

    c = fda.Control(s)

    phi_pvd = fd.File("phi_evolution_euclid.pvd", target_continuity=fd.H1)

    def deriv_cb(phi):
        with stop_annotating():
            phi_pvd.write(phi[0])

    Jhat = LevelSetFunctional(J, c, phi, derivative_cb_pre=deriv_cb)
    Vhat = LevelSetFunctional(Vol, c, phi)

    bcs_vel_1 = fd.DirichletBC(S, noslip, (1, 2, 3, 4))
    bcs_vel = [bcs_vel_1]
    reg_solver = RegularizationSolver(
        S, mesh, beta=0.5, gamma=1e5, dx=dx, bcs=bcs_vel, design_domain=0
    )

    tol = 1e-5
    dt = 0.0002
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
        ineqconstraints=[Constraint(Vhat, Vval, VControl)],
    )
    _ = nlspace_solve_shape(problem, params)


if __name__ == "__main__":
    main()
