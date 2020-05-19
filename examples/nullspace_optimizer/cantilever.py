from firedrake import *
from firedrake_adjoint import *

from lestofire import (
    LevelSetLagrangian,
    RegularizationSolver,
    HJStabSolver,
    SignedDistanceSolver,
    EuclideanOptimizable,
    nlspace_solve_shape
)

from pyadjoint import no_annotations

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

Jform = assemble(inner(hs(-phi, beta)*sigma(u_sol), epsilon(u_sol))*dx)
#Jform = assemble(inner(hs(-phi, beta)*sigma(u_sol), epsilon(u_sol))*dx + Constant(1000.0)*hs(-phi, beta)*dx)
VolPen = assemble(hs(-phi, beta)*dx)
VolControl = Control(VolPen)

Vval = 1.0

phi_pvd = File("phi_evolution.pvd")
beta1_pvd = File("beta1.pvd")
beta2_pvd = File("beta2.pvd")
newvel_pvd = File("newvel.pvd")
newvel = Function(S)
newphi = Function(PHI)


velocity = Function(S)
bcs_vel_1 = DirichletBC(S, Constant((0.0, 0.0)), 1)
bcs_vel_2 = DirichletBC(S, Constant((0.0, 0.0)), 2)
bcs_vel = [bcs_vel_1, bcs_vel_2]

c = Control(s)
Jhat = LevelSetLagrangian(Jform, c, phi)
Vhat = LevelSetLagrangian(VolPen, c, phi)
beta_param = 1e0
reg_solver = RegularizationSolver(S, mesh, beta=beta_param, gamma=1.0e5, dx=dx, bcs=bcs_vel, output_dir=None)
reinit_solver = SignedDistanceSolver(mesh, PHI, dt=1e-7, iterative=False)
hj_solver = HJStabSolver(mesh, PHI, c2_param=1.0, iterative=False)
#dt = 0.5*1e-1
dt = 10.0
tol = 1e-5

class InfDimProblem(EuclideanOptimizable):
    def __init__(self, phi, Jhat, Hhat, Hval, Hcontrol, control):
        super().__init__(1) # This argument is the number of variables, it doesn't really matter...
        self.nconstraints = 0
        self.nineqconstraints = 1
        self.V = control.control.function_space()
        self.dJ = Function(self.V)
        self.dH = Function(self.V)
        self.dx = Function(self.V)
        self.Jhat = Jhat

        self.Hhat = Hhat
        self.Hval = Hval
        self.Hcontrol = Hcontrol

        self.phi = phi
        self.control = control.control
        self.newphi = Function(phi.function_space())
        self.i = 0 # iteration count

    def fespace(self):
        return self.V

    def x0(self):
        return self.phi

    def J(self, x):
        return self.Jhat(x)

    def dJT(self, x):
        dJ = self.Jhat.derivative()
        reg_solver.solve(self.dJ, dJ)
        beta1_pvd.write(self.dJ)
        return self.dJ

    def H(self, x):
        return [self.Hcontrol.tape_value() - self.Hval]

    def dHT(self, x):
        dH = self.Hhat.derivative()
        reg_solver.solve(self.dH, dH)
        beta2_pvd.write(self.dH)
        return [self.dH]

    @no_annotations
    def reinit(self, x):
        if self.i % 10 == 0:
            Dx = 0.01
            x.assign(reinit_solver.solve(x, Dx), annotate=False)

    def eval_gradients(self, x):
        """Returns the triplet (dJT(x),dGT(x),dHT(x))
        Is used by nslpace_solve method only if self.inner_product returns
        None"""
        self.i += 1
        newphi.assign(x)
        phi_pvd.write(newphi)

        dJT = self.dJT(x)
        if self.nconstraints == 0:
            dGT = []
        else:
            dGT = self.dGT(x)
        if self.nineqconstraints == 0:
            dHT = []
        else:
            dHT = self.dHT(x)
        return (dJT, dGT, dHT)

    def retract(self, x, dx):
        import numpy as np
        maxv = np.max(x.vector()[:])
        hmin = 0.2414
        dt = 0.1 * 1.0 * hmin / maxv
        #dt = 0.01
        self.newphi.assign(hj_solver.solve(Constant(-1.0)*dx, x, steps=1, dt=dt), annotate=False)
        newvel.assign(dx, annotate=False)
        newvel_pvd.write(newvel)
        return self.newphi

    @no_annotations
    def inner_product(self, x, y):
        #return assemble(beta_param*inner(grad(x), grad(y))*dx + inner(x, y)*dx)
        return assemble(inner(x, y)*dx)


parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "mat_type": "aij",
    "ksp_converged_reason": None,
    "pc_factor_mat_solver_type": "mumps",
}

params = {"alphaC": 0.5, "debug": 5, "alphaJ": 1.0, "dt": dt, "maxtrials": 10, "itnormalisation" : 1, "tol" : tol}
results = nlspace_solve_shape(InfDimProblem(phi, Jhat, Vhat, Vval, VolControl, c), params)
