from firedrake import *
from firedrake_adjoint import *
import numpy as np
from ufl import replace, conditional
from ufl import min_value, max_value

from mesh_stokes_flow import INMOUTH1, INMOUTH2, OUTMOUTH1, OUTMOUTH2, WALLS

from lestofire.levelset import LevelSetFunctional, RegularizationSolver
from lestofire.optimization import HJLocalDG, ReinitSolverDG
from nullspace_optimizer.lestofire import nlspace_solve_shape, Constraint, InfDimProblem
from pyadjoint import no_annotations, stop_annotating


def main():
    output_dir = "stokes_levelset_darcy/"

    mesh = Mesh("./mesh_stokes.msh")

    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    x, y = SpatialCoordinate(mesh)
    PHI = FunctionSpace(mesh, "CG", 1)
    lx = 2.0
    ly = 1.0
    phi_expr = -cos(6.0 / lx * pi * x + 1.0) * cos(4.0 * pi * y) - 0.6

    with stop_annotating():
        phi = interpolate(phi_expr, PHI)
    phi.rename("LevelSet")
    with stop_annotating():
        File(output_dir + "phi_initial.pvd").write(phi)

    mu = Constant(1e2)
    alphamax = 1e6
    alphamin = 1e-12
    epsilon = Constant(100000.0)

    def hs(phi, epsilon):
        return Constant(1.0) / (Constant(1.0) + exp(-epsilon * phi)) + Constant(
            alphamin
        )

    P2 = VectorElement("CG", mesh.ufl_cell(), 2)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = FunctionSpace(mesh, TH)

    U = Function(W, name="Solution")
    u, p = split(U)
    V = TestFunction(W)
    v, q = split(V)

    f = Constant((0.0, 0.0))
    a = mu * inner(grad(u), grad(v)) - div(v) * p - q * div(u)
    darcy_term = Constant(alphamax) * hs(phi, epsilon) * inner(u, v)
    e1f = a * dx + darcy_term * dx

    L = inner(f, v) * dx

    inflow = Constant((5.0, 0.0))
    noslip = Constant((0.0, 0.0))
    bc1 = DirichletBC(W.sub(0), noslip, WALLS)
    bc2 = DirichletBC(W.sub(0), inflow, 1)
    bc3 = DirichletBC(W.sub(0), inflow, 2)
    bc4 = DirichletBC(W.sub(0), inflow, 3)
    bc5 = DirichletBC(W.sub(0), inflow, 4)
    bcs = [bc1, bc2, bc3, bc4, bc5]

    parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "mat_type": "aij",
        "ksp_converged_reason": None,
        "pc_factor_mat_solver_type": "mumps",
    }
    a = derivative(e1f, U)
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

    solve(a == L, U, bcs, solver_parameters=parameters, nullspace=nullspace)
    u, p = split(U)

    Vol = assemble(hs(-phi, epsilon) * Constant(1.0) * dx(domain=mesh))
    VControl = Control(Vol)
    Vval = assemble(Constant(0.3) * dx(domain=mesh), annotate=False)
    with stop_annotating():
        print("Initial constraint function value {}".format(Vol))

    J = assemble(
        Constant(alphamax) * hs(phi, epsilon) * inner(u, u) * dx
        + mu / Constant(2.0) * inner(grad(u), grad(u)) * dx
    )

    c = Control(s)

    phi_pvd = File("phi_evolution.pvd")

    def deriv_cb(phi):
        with stop_annotating():
            phi_pvd.write(phi[0])

    Jhat = LevelSetFunctional(J, c, phi, derivative_cb_pre=deriv_cb)
    Vhat = LevelSetFunctional(Vol, c, phi)

    velocity = Function(S)
    bcs_vel_1 = DirichletBC(S, noslip, 1)
    bcs_vel_2 = DirichletBC(S, noslip, 2)
    bcs_vel_3 = DirichletBC(S, noslip, 3)
    bcs_vel_4 = DirichletBC(S, noslip, 4)
    bcs_vel_5 = DirichletBC(S, noslip, 5)
    bcs_vel = [bcs_vel_1, bcs_vel_2, bcs_vel_3, bcs_vel_4, bcs_vel_5]
    reg_solver = RegularizationSolver(
        S, mesh, beta=1, gamma=0.0, dx=dx, bcs=bcs_vel, output_dir=None
    )

    reinit_solver = ReinitSolver(mesh, PHI, dt=1e-7, iterative=False)
    hj_solver = HJStabSolver(mesh, PHI, c2_param=0.5, iterative=False)
    # dt = 0.5*1e-1
    dt = 1.0
    tol = 1e-5

    vol_constraint = Constraint(Vhat, Vval, VControl)
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
        "maxit": 85,
        "debug": 5,
        "alphaJ": 1.0,
        "dt": dt,
        "maxtrials": 10,
        "itnormalisation": 50,
        "tol": tol,
    }
    results = nlspace_solve_shape(problem, params)


if __name__ == "__main__":
    main()
