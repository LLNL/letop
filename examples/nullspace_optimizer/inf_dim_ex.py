from firedrake import *
from firedrake_adjoint import *
from lestofire import EuclideanOptimizable, nlspace_solve_shape

N = 500
mesh = UnitSquareMesh(N, N, quadrilateral=True)
V = FunctionSpace(mesh, 'DG', 0)
x = Function(V)
x.assign(Constant(1.0))
coords = SpatialCoordinate(mesh)
b = project(coords[0] * coords[1], V, annotate=False)
#b = Function(V)
#b.assign(Constant(2.0), annotate=False)

J = assemble(x*x*dx)
m = Control(x)
Jhat = ReducedFunctional(J, m)
Jhat.derivative()
G = assemble(-x*b*dx)
Ghat = ReducedFunctional(G, m)
Gval = -5
#dt = 1.0 / N * 1e1
dt = 0.5

class InfDimProblem(EuclideanOptimizable):
    def __init__(self, Jhat, Ghat, G, control):
        super().__init__(1) # This argument is the number of variables, it doesn't really matter...
        self.nconstraints = 0
        self.nineqconstraints = 1
        self.V = control.control.function_space()
        self.dJ = Function(self.V)
        self.dh1 = Function(self.V)
        self.dh2 = Function(self.V)
        self.dx = Function(self.V)
        self.Jhat = Jhat
        self.Ghat = Ghat
        self.Gval = G
        self.control = control.control
        self.newx = Function(self.V)

    def fespace(self):
        return self.V

    def x0(self):
        return self.control

    def J(self, x):
        return self.Jhat(x)

    def dJT(self, x):
        return self.Jhat.derivative(options={"riesz_representation": "L2"})

    def H(self, x):
        return [self.Ghat(x) - self.Gval]

    def dHT(self, x):
        return [self.Ghat.derivative(options={"riesz_representation": "L2"})]

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
        self.newx.assign(x + dx, annotate=False)
        return self.newx


params = {"alphaC": 0.5, "debug": 5, "alphaJ": 0.5, "dt": dt, "maxtrials": 20, "itnormalisation" : 2, "tol" : 1e-4}
results = nlspace_solve_shape(InfDimProblem(Jhat, Ghat, Gval, m), params)
File("solution.pvd").write(results['x'][-1])
