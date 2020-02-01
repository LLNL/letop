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
    phi_expr = sin(y*pi/2.0)*cos(x*pi/2.0) - Constant(0.8)
    phi = interpolate(phi_expr , PHI)
    phi.rename("LevelSet")
    File(output_dir + "phi_initial.pvd").write(phi)

    mu = Constant(1e2)
    rho_min = 1e-5
    beta = Constant(100000.0)
    def hs(phi, beta):
        return Constant(1.0) / (Constant(1.0) + exp(-beta*phi)) + Constant(rho_min)

    H1 = VectorElement("CG", mesh.ufl_cell(), 2)
    W = FunctionSpace(mesh, H1)

    u = TrialFunction(W)
    v = TestFunction(W)

    # Elasticity parameters
    E, nu = 10.0, 0.3
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

    def epsilon(u):
        return sym(nabla_grad(u))

    def sigma(v):
        return 2.0*mu*epsilon(v) + lmbda*tr(epsilon(v))*Identity(2)

    f = Constant((0.0, 0.0))
    a = inner(hs(phi, beta)*sigma(u), grad(v))*dx
    t = Constant((0.0, -5.0))
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

    J = inner(t, u_sol)*ds(2)

    Jform = assemble(J)

    phi_pvd = File(output_dir + "phi_evolution.pvd")
    def deriv_cb(phi):
        phi_pvd.write(phi[0])

    c = Control(s)
    Jhat = LevelSetLagrangian(Jform, c, phi, derivative_cb_pre=deriv_cb)
    Jhat_v = Jhat(phi)

    velocity = Function(S)
    bcs_vel_1 = DirichletBC(S, Constant((0.0, 0.0)), 1)
    bcs_vel = [bcs_vel_1]
    reg_solver = RegularizationSolver(S, mesh, beta=1e4, gamma=0.0, dx=dx, bcs=bcs_vel)

    options = {
             'hmin' : 0.001414,
             'hj_stab': 2.0,
             'dt_scale' : 0.1,
             'n_hj_steps' : 3,
             'n_reinit' : 5,
             'max_iter' : 40
             }

    opti_solver = SteepestDescent(Jhat, reg_solver, options=options, pvd_output=phi_pvd)
    Jarr = opti_solver.solve(phi, velocity, iterative=False, tolerance=1e-4)

if __name__ == '__main__':
    main()
