import firedrake as fd
from lestofire.physics import AdvectionDiffusionGLS
from solver_parameters import temperature_solver_parameters


def temperature_solver(T, t, beta, PeInv, t1, INLET1, t2, INLET2):
    F_T = AdvectionDiffusionGLS(T, beta, t, PeInv=PeInv)

    bc1 = fd.DirichletBC(T, t1, INLET1)
    bc2 = fd.DirichletBC(T, t2, INLET2)
    bcs = [bc1, bc2]
    problem_T = fd.NonlinearVariationalProblem(F_T, t, bcs=bcs)
    return fd.NonlinearVariationalSolver(
        problem_T, solver_parameters=temperature_solver_parameters
    )
