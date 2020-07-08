from firedrake import *
from firedrake_adjoint import *
import numpy as np
from ufl import replace, conditional
from ufl import min_value, max_value

from lestofire import LevelSetLagrangian, SteepestDescent, RegularizationSolver

def main():
    output_dir = "cantilever/"

    mesh = Mesh("./mesh_cantilever.msh")

    S = VectorFunctionSpace(mesh, "CG", 1)
    s = Function(S,name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    x, y = SpatialCoordinate(mesh)
    PHI = FunctionSpace(mesh, 'CG', 1)
    lx = 2.0
    ly = 1.0
    phi_expr = -cos(6.0/lx*pi*x) * cos(4.0*pi*y) - 0.6\
                + max_value(200.0*(0.01-x**2-(y-ly/2)**2),.0)\
                + max_value(100.0*(x+y-lx-ly+0.1),.0) + max_value(100.0*(x-y-lx+0.1),.0)
    phi = interpolate(phi_expr , PHI)
    phi.rename("LevelSet")
    File(output_dir + "phi_initial.pvd").write(phi)



    rho_min = 1e-3
    beta = Constant(1000.0)
    def hs(phi, beta):
        return Constant(1.0) / (Constant(1.0) + exp(-beta*phi)) + Constant(rho_min)

    H1 = VectorElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, H1)

    u = TrialFunction(W)
    v = TestFunction(W)

    # Elasticity parameters
    E, nu = 1.0, 0.3
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

    def epsilon(u):
        return sym(nabla_grad(u))

    def sigma(v):
        return 2.0*mu*epsilon(v) + lmbda*tr(epsilon(v))*Identity(2)

    f = Constant((0.0, 0.0))
    a = inner(hs(-phi, beta)*sigma(u), nabla_grad(v))*dx
    t = Constant((0.0, -75.0))
    L = inner(t, v)*ds(2)

    bc = DirichletBC(W, Constant((0.0, 0.0)), 1)
    parameters = {
            'ksp_type':'preonly', 'pc_type':'lu',
            "mat_type": "aij",
            'ksp_converged_reason' : None,
            "pc_factor_mat_solver_type": "mumps"
            }
    u_sol = Function(W)
    solve(a==L, u_sol, bcs=[bc], solver_parameters=parameters)#, nullspace=nullspace)
    File("u_sol.pvd").write(u_sol)
    Sigma = TensorFunctionSpace(mesh, 'CG', 1)
    File("sigma.pvd").write(project(sigma(u_sol), Sigma))

    J = inner(hs(-phi, beta)*sigma(u_sol), epsilon(u_sol))*dx + Constant(1000.0)*hs(-phi, beta)*dx

    Jform = assemble(J)
    print("Compliance: {:2f}".format(Jform))

    phi_pvd = File(output_dir + "phi_evolution_uniq.pvd")

    c = Control(s)
    Jhat = LevelSetLagrangian(Jform, c, phi)
    Jhat.optimize_tape()
    Jhat_v = Jhat(phi)

    velocity = Function(S)
    bcs_vel_1 = DirichletBC(S, Constant((0.0, 0.0)), 1)
    bcs_vel_2 = DirichletBC(S, Constant((0.0, 0.0)), 2)
    bcs_vel = [bcs_vel_1, bcs_vel_2]
    reg_solver = RegularizationSolver(S, mesh, beta=5e0, gamma=1e5, dx=dx, bcs=bcs_vel)

    options = {
             'hmin' : 0.02414,
             'hj_stab': 2.0,
             'dt_scale' : 0.1,
             'n_hj_steps' : 1,
             'n_reinit' : 5,
             'max_iter' : 200
             }

    opti_solver = SteepestDescent(Jhat, reg_solver, options=options, pvd_output=phi_pvd)
    Jarr = opti_solver.solve(phi, velocity, iterative=False, tolerance=1e-10)

if __name__ == '__main__':
    main()
