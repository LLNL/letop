from firedrake import *
from firedrake_adjoint import *

from lestofire import (
    LevelSetLagrangian,
    RegularizationSolver,
    HJStabSolver,
    SignedDistanceSolver,
    EuclideanOptimizable,
    nlspace_solve_shape,
)

from params import (
    INMOUTH2,
    INMOUTH1,
    line_sep,
    dist_center,
    inlet_width,
    WALLS,
    INLET1,
    INLET2,
    OUTLET1,
    OUTLET2,
    width,
    OUTMOUTH1,
    OUTMOUTH2,
)

from pyadjoint import no_annotations

from draw import drawMuls, drawJ, drawC

output_dir = "2D/"

mesh = Mesh("./2D_mesh.msh")

S = VectorFunctionSpace(mesh, "CG", 1)
s = Function(S, name="deform")
mesh.coordinates.assign(mesh.coordinates + s)

x, y = SpatialCoordinate(mesh)
PHI = FunctionSpace(mesh, "CG", 1)
phi_expr = sin(y * pi / 0.2) * cos(x * pi / 0.2) - Constant(0.8)
phi = interpolate(phi_expr, PHI)
phi.rename("LevelSet")
File(output_dir + "phi_initial.pvd").write(phi)


# Parameters
mu = Constant(0.08)  # viscosity
alphamin = 1e-12
alphamax = 2.5 / (2e-5)
parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    # "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
stokes_parameters = parameters
temperature_parameters = parameters
epsilon = Constant(10000.0)
u_inflow = 2e-3
tin1 = Constant(10.0)
tin2 = Constant(100.0)

Re = u_inflow * width / mu.values()[0]
print("Reynolds number: {:.5f}".format(Re), flush=True)


P2 = VectorElement("CG", mesh.ufl_cell(), 2)
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

U = TrialFunction(W)
u, p = split(U)
V = TestFunction(W)
v, q = split(V)


def hs(phi, epsilon):
    return Constant(alphamax) * Constant(1.0) / (
        Constant(1.0) + exp(-epsilon * phi)
    ) + Constant(alphamin)


def stokes(phi, BLOCK_INLET_MOUTH, BLOCK_OUTLET_MOUTH):
    a_fluid = mu * inner(grad(u), grad(v)) - div(v) * p - q * div(u)
    darcy_term = inner(u, v)
    return (
        a_fluid * dx
        + hs(phi, epsilon) * darcy_term * dx(0)
        + alphamax * darcy_term * (dx(BLOCK_INLET_MOUTH) + dx(BLOCK_OUTLET_MOUTH))
    )


# Dirichelt boundary conditions
inflow1 = as_vector(
    [
        u_inflow
        * sin(((y - (line_sep - (dist_center + inlet_width))) * pi) / inlet_width),
        0.0,
    ]
)
inflow2 = as_vector(
    [u_inflow * sin(((y - (line_sep + dist_center)) * pi) / inlet_width), 0.0]
)

noslip = Constant((0.0, 0.0))

# Stokes 1
bcs1_1 = DirichletBC(W.sub(0), noslip, WALLS)
bcs1_2 = DirichletBC(W.sub(0), inflow1, INLET1)
bcs1_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET1)
bcs1_4 = DirichletBC(W.sub(0), noslip, INLET2)
bcs1_5 = DirichletBC(W.sub(0), noslip, OUTLET2)
bcs1 = [bcs1_1, bcs1_2, bcs1_3, bcs1_4, bcs1_5]

# Stokes 2
bcs2_1 = DirichletBC(W.sub(0), noslip, WALLS)
bcs2_2 = DirichletBC(W.sub(0), inflow2, INLET2)
bcs2_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET2)
bcs2_4 = DirichletBC(W.sub(0), noslip, INLET1)
bcs2_5 = DirichletBC(W.sub(0), noslip, OUTLET1)
bcs2 = [bcs2_1, bcs2_2, bcs2_3, bcs2_4, bcs2_5]

# Forward problems
U1, U2 = Function(W), Function(W)
L = inner(Constant((0.0, 0.0, 0.0)), V) * dx
problem = LinearVariationalProblem(stokes(-phi, INMOUTH2, OUTMOUTH2), L, U1, bcs=bcs1)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
solver_stokes1 = LinearVariationalSolver(
    problem, solver_parameters=stokes_parameters
)  # , nullspace=nullspace)
solver_stokes1.solve()
problem = LinearVariationalProblem(stokes(phi, INMOUTH1, OUTMOUTH1), L, U2, bcs=bcs2)
solver_stokes2 = LinearVariationalSolver(problem, solver_parameters=stokes_parameters)
solver_stokes2.solve()

# Convection difussion equation
ks = Constant(1e0)
cp_value = 5.0e5
cp = Constant(cp_value)
T = FunctionSpace(mesh, "DG", 1)
t = Function(T, name="Temperature")
w = TestFunction(T)

# Mesh-related functions
n = FacetNormal(mesh)
h = CellDiameter(mesh)
u1, p1 = split(U1)
u2, p2 = split(U2)


def upwind(u):
    return (dot(u, n) + abs(dot(u, n))) / 2.0


u1n = upwind(u1)
u2n = upwind(u2)

# Penalty term
alpha = Constant(
    50000.0
)  # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
alpha = Constant(
    500.0
)  # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
# Bilinear form
a_int = dot(grad(w), ks * grad(t) - cp * (u1 + u2) * t) * dx

a_fac = (
    Constant(-1.0) * ks * dot(avg(grad(w)), jump(t, n)) * dS
    + Constant(-1.0) * ks * dot(jump(w, n), avg(grad(t))) * dS
    + ks("+") * (alpha("+") / avg(h)) * dot(jump(w, n), jump(t, n)) * dS
)

a_vel = (
    dot(
        jump(w),
        cp * (u1n("+") + u2n("+")) * t("+") - cp * (u1n("-") + u2n("-")) * t("-"),
    )
    * dS
    + dot(w, cp * (u1n + u2n) * t) * ds
)

a_bnd = (
    dot(w, cp * dot(u1 + u2, n) * t) * (ds(INLET1) + ds(INLET2))
    + w * t * (ds(INLET1) + ds(INLET2))
    - w * tin1 * ds(INLET1)
    - w * tin2 * ds(INLET2)
    + alpha / h * ks * w * t * (ds(INLET1) + ds(INLET2))
    - ks * dot(grad(w), t * n) * (ds(INLET1) + ds(INLET2))
    - ks * dot(grad(t), w * n) * (ds(INLET1) + ds(INLET2))
)

aT = a_int + a_fac + a_vel + a_bnd

LT_bnd = (
    alpha / h * ks * tin1 * w * ds(INLET1)
    + alpha / h * ks * tin2 * w * ds(INLET2)
    - tin1 * ks * dot(grad(w), n) * ds(INLET1)
    - tin2 * ks * dot(grad(w), n) * ds(INLET2)
)
eT = aT - LT_bnd

problem = NonlinearVariationalProblem(eT, t)
solver_temp = NonlinearVariationalSolver(
    problem, solver_parameters=temperature_parameters
)
solver_temp.solve()

power_drop = 1e-2
Power1 = assemble(p1 / power_drop * ds(INLET1))
Power2 = assemble(p2 / power_drop * ds(INLET2))
Power = Power1 + Power2
scale_factor = 10
Jform = assemble(Constant(-scale_factor * cp_value) * inner(t * u1, n) * ds(OUTLET1))

U1control = Control(U1)
U2control = Control(U2)
u1_pvd = File(output_dir + "u1.pvd")
u2_pvd = File(output_dir + "u2.pvd")
tcontrol = Control(t)
tplot = Function(T)
t_pvd = File(output_dir + "t.pvd")


def deriv_cb(phi):
    with stop_annotating():
        u1, _ = U1control.tape_value().split()
        u2, _ = U2control.tape_value().split()
        tplot.assign(tcontrol.tape_value())
    u1.rename("Velocity")
    u2.rename("Velocity")
    u1_pvd.write(u1)
    u2_pvd.write(u2)
    t_pvd.write(tplot)


c = Control(s)

# Reduced Functionals
Jhat = LevelSetLagrangian(Jform, c, phi, derivative_cb_pre=deriv_cb)
P1hat = LevelSetLagrangian(Power1, c, phi, derivative_cb_pre=deriv_cb)
P1control = Control(Power1)

P2hat = LevelSetLagrangian(Power2, c, phi, derivative_cb_pre=deriv_cb)
P2control = Control(Power2)

Jhat_v = Jhat(phi)
print("Initial cost function value {:.5f}".format(Jhat_v), flush=True)
print("Power drop 1 {:.5f}".format(Power1), flush=True)
print("Power drop 2 {:.5f}".format(Power2), flush=True)

# Jhat.optimize_tape()

velocity = Function(S)
bcs_vel_1 = DirichletBC(S, noslip, 1)
bcs_vel_2 = DirichletBC(S, noslip, 2)
bcs_vel_3 = DirichletBC(S, noslip, 3)
bcs_vel_4 = DirichletBC(S, noslip, 4)
bcs_vel_5 = DirichletBC(S, noslip, 5)
bcs_vel = [bcs_vel_1, bcs_vel_2, bcs_vel_3, bcs_vel_4, bcs_vel_5]
reg_solver = RegularizationSolver(
    S, mesh, beta=5e1, gamma=1e5, dx=dx, sim_domain=0, output_dir=None
)


reinit_solver = SignedDistanceSolver(mesh, PHI, dt=1e-7, iterative=False)
hj_solver = HJStabSolver(mesh, PHI, c2_param=1.0, iterative=False)
# dt = 0.5*1e-1
dt = 5.0
tol = 1e-5

phi_pvd = File("phi_evolution.pvd")
beta1_pvd = File("beta1.pvd")
beta2_pvd = File("beta2.pvd")
newvel_pvd = File("newvel.pvd")
newvel = Function(S)
newphi = Function(PHI)


class InfDimProblem(EuclideanOptimizable):
    def __init__(
        self, phi, Jhat, H1hat, H1val, H1control, H2hat, H2val, H2control, control
    ):
        super().__init__(
            1
        )  # This argument is the number of variables, it doesn't really matter...
        self.nconstraints = 0
        self.nineqconstraints = 2
        self.V = control.control.function_space()
        self.dJ = Function(self.V)
        self.dH1 = Function(self.V)
        self.dH2 = Function(self.V)
        self.dx = Function(self.V)
        self.Jhat = Jhat

        self.H1hat = H1hat
        self.H1val = H1val
        self.H1control = H1control

        self.H2hat = H2hat
        self.H2val = H2val
        self.H2control = H2control

        self.phi = phi
        self.control = control.control
        self.newphi = Function(phi.function_space())
        self.i = 0  # iteration count

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
        return [self.H1control.tape_value() - self.H1val, self.H2control.tape_value() - self.H2val]

    def dHT(self, x):
        dH = self.H1hat.derivative()
        reg_solver.solve(self.dH1, dH)
        dH = self.H2hat.derivative()
        reg_solver.solve(self.dH2, dH)
        return [self.dH1, self.dH2]

    @no_annotations
    def reinit(self, x):
        if self.i % 10 == 0:
            Dx = 0.01
            x.assign(reinit_solver.solve(x, Dx), annotate=False)

    @no_annotations
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

    @no_annotations
    def retract(self, x, dx):
        import numpy as np

        maxv = np.max(x.vector()[:])
        hmin = 0.00940  # Hard coded from FEniCS
        dt = 0.1 * 1.0 * hmin / maxv
        # dt = 0.01
        self.newphi.assign(
            hj_solver.solve(Constant(-1.0) * dx, x, steps=1, dt=dt), annotate=False
        )
        newvel.assign(dx, annotate=False)
        newvel_pvd.write(newvel)
        return self.newphi

    @no_annotations
    def inner_product(self, x, y):
        # return assemble(beta_param*inner(grad(x), grad(y))*dx + inner(x, y)*dx)
        return assemble(inner(x, y) * dx)


params = {
    "alphaC": 1.0,
    "debug": 5,
    "alphaJ": 0.5,
    "dt": dt,
    #"K": 100,
    "maxit": 200,
    "maxtrials": 10,
    "itnormalisation": 50,
    "tol": tol,
}
results = nlspace_solve_shape(
    InfDimProblem(phi, Jhat, Phat, 0.0, P1control, P2control, c), params
)

import matplotlib.pyplot as plt
plt.figure()
drawMuls(results, 'NLSPACE')
plt.legend()
plt.savefig(
    "muls.pdf",
    dpi=1600,
    orientation="portrait",
    papertype=None,
    format=None,
    transparent=True,
    bbox_inches="tight",
)

plt.figure()
drawJ(results)
plt.legend()
plt.savefig(
    "J.pdf",
    dpi=1600,
    orientation="portrait",
    papertype=None,
    format=None,
    transparent=True,
    bbox_inches="tight",
)

plt.figure()
drawC(results)
plt.legend()
plt.savefig(
    "C.pdf",
    dpi=1600,
    orientation="portrait",
    papertype=None,
    format=None,
    transparent=True,
    bbox_inches="tight",
)

plt.show()
