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


mesh = UnitSquareMesh(100, 100)

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="deform")
mesh.coordinates.assign(mesh.coordinates + s)

x, y = SpatialCoordinate(mesh)
PHI = FunctionSpace(mesh, "CG", 1)
phi_expr = sin(y * pi / 0.2) * cos(x * pi / 0.2) - Constant(0.8)

phi = interpolate(phi_expr, PHI)
phi.rename("LevelSet")

alphamin = 1e-12
epsilon = Constant(100000.0)


def hs(phi, epsilon):
    return Constant(1.0) / (Constant(1.0) + exp(-epsilon * phi)) + Constant(alphamin)


scaling = -100.0
Jform = assemble(Constant(scaling) * hs(phi, epsilon) * x * dx)
print("Initial cost function value {}".format(Jform))
VolPen = assemble(hs(phi, epsilon) * dx(domain=mesh))
Vval = 0.5

phi_pvd = File("phi_evolution.pvd")
beta1_pvd = File("beta1.pvd")
beta2_pvd = File("beta2.pvd")
newvel_pvd = File("newvel.pvd")
newvel = Function(S)


def deriv_cb(phi):
    phi_pvd.write(phi[0])


c = Control(s)
Jhat = LevelSetLagrangian(Jform, c, phi)
Vhat = LevelSetLagrangian(VolPen, c, phi)
reg_solver = RegularizationSolver(S, mesh, beta=1e-2, gamma=1.0e0, dx=dx, output_dir=None)
hj_solver = HJStabSolver(mesh, PHI, c2_param=1.0, iterative=False)
dt = 0.5*1e0
tol = 1e-5

class InfDimProblem(EuclideanOptimizable):
    def __init__(self, phi, Jhat, Hhat, G, control):
        super().__init__(1) # This argument is the number of variables, it doesn't really matter...
        self.nconstraints = 0
        self.nineqconstraints = 1
        self.V = control.control.function_space()
        self.dJ = Function(self.V)
        self.dH = Function(self.V)
        self.dx = Function(self.V)
        self.Jhat = Jhat
        self.Hhat = Hhat
        self.Hval = G
        self.phi = phi
        self.control = control.control
        self.newphi = Function(phi.function_space())

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
        return [self.Hhat(x) - self.Hval]

    def dHT(self, x):
        dH = self.Hhat.derivative()
        reg_solver.solve(self.dH, dH)
        beta2_pvd.write(self.dH)
        return [self.dH]

    def eval_gradients(self, x):
        """Returns the triplet (dJT(x),dGT(x),dHT(x))
        Is used by nslpace_solve method only if self.inner_product returns
        None"""
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
        dt = 0.1
        self.newphi.assign(hj_solver.solve(Constant(-1.0)*dx, x, steps=1, dt=dt), annotate=False)
        phi_pvd.write(self.newphi)
        newvel.assign(dx, annotate=False)
        newvel_pvd.write(newvel)
        return self.newphi

options = {
    "hmin": 0.01414,
    "hj_stab": 5.0,
    "dt_scale": 1e-2,
    "n_hj_steps": 3,
    "max_iter": 30,
    "n_reinit": 5,
    "stopping_criteria": 1e-2,
}

parameters = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "mat_type": "aij",
    "ksp_converged_reason": None,
    "pc_factor_mat_solver_type": "mumps",
}

params = {"alphaC": 0.5, "debug": 5, "alphaJ": 0.5, "dt": dt, "maxtrials": 20, "itnormalisation" : 20, "tol" : tol}
results = nlspace_solve_shape(InfDimProblem(phi, Jhat, Vhat, Vval, c), params)

velocity = Function(S)
