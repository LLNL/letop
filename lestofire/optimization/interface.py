from lestofire import (
    EuclideanOptimizable,
    LevelSetLagrangian,
    RegularizationSolver,
    HJStabSolver,
    SignedDistanceSolver,
)
from pyadjoint import Control, no_annotations
from pyadjoint.enlisting import Enlist

from firedrake import Function, assemble, inner, dx, Constant


class Constraint(object):
    def __init__(self, rfnl, constraint_value, scalar_control):
        """Docstring for __init__.

        :rfnl: LevelSetFunctional
        :constraint_value: Scalar
        :scalar_control: Control

        """
        if not isinstance(rfnl, LevelSetLagrangian):
            raise TypeError(
                f"Provided '{type(rfnl).__name__}', not a LevelSetFunctional"
            )
        if not isinstance(scalar_control, Control):
            raise TypeError(
                f"Provided '{type(scalar_control).__name__}', not a Control"
            )
        if not isinstance(constraint_value, float):
            raise TypeError(
                f"Provided '{type(constraint_value).__name__}', not a float"
            )

        self.rfnl = rfnl
        self.constraint_value = constraint_value
        self.scalar_control = scalar_control

    def evaluate(self, x):
        # TODO: Any way to check that the value saved in self.scalar_control corresponds
        # corresponds to `x`?
        return self.scalar_control.tape_value() - self.constraint_value

    def derivative(self, x):
        return self.rfnl.derivative(x)


class InfDimProblem(EuclideanOptimizable):
    def __init__(
        self,
        phi,
        cost_function,
        reg_solver,
        hj_solver,
        reinit_solver,
        eqconstraint=None,
        ineqconstraints=None,
    ):
        super().__init__(
            1
        )  # This argument is the number of variables, it doesn't really matter...
        if not isinstance(reg_solver, RegularizationSolver):
            raise TypeError(
                f"Provided regularization solver '{type(reg_solver).__name__}', is not a RegularizationSolver"
            )
        self.reg_solver = reg_solver
        if not isinstance(hj_solver, HJStabSolver):
            raise TypeError(
                f"Provided Hamilton-Jacobi solver '{type(hj_solver).__name__}', is not a HJStabSolver"
            )
        self.hj_solver = hj_solver
        if not isinstance(reinit_solver, SignedDistanceSolver):
            raise TypeError(
                f"Provided reinitialization solver '{type(hj_solver).__name__}', is not a SignedDistanceSolver"
            )
        self.reinit_solver = reinit_solver

        self.eqconstraint = Enlist(eqconstraint)
        self.ineqconstraints = Enlist(ineqconstraints)
        for constr in self.eqconstraint:
            if not isinstance(constr, Constraint):
                raise TypeError(
                    f"Provided equality constraint '{type(constr).__name__}', not a Constraint"
                )
        for ineqconstr in self.ineqconstraints:
            if not isinstance(ineqconstr, Constraint):
                raise TypeError(
                    f"Provided inequality constraint '{type(ineqconstr).__name__}', not a Constraint"
                )
        self.nineqconstraints = Enlist(ineqconstraints)

        assert len(cost_function.controls) < 2, "Only one control for now"
        self.V = cost_function.controls[0].control.function_space()

        self.gradJ = Function(self.V)
        self.gradH = [Function(self.V) for _ in self.ineqconstraints]
        self.gradG = [Function(self.V) for _ in self.eqconstraints]

        self.dx = Function(self.V)
        self.cost_function = cost_function

        self.phi = phi
        self.newphi = Function(phi.function_space())
        self.i = 0  # iteration count

    def fespace(self):
        return self.V

    def x0(self):
        return self.phi

    def J(self, x):
        return self.cost_function(x)

    def dJ(self, x):
        return self.Jhat.derivative()

    def G(self, x):
        return [eqconstr.evaluate(x) for eqconstr in self.eqconstraints]

    def dG(self, x):
        return [eqconstr.derivative(x) for eqconstr in self.eqconstraints]

    def H(self, x):
        return [ineqconstr.evaluate(x) for ineqconstr in self.ineqconstraints]

    def dH(self, x):
        return [ineqconstr.derivative(x) for ineqconstr in self.ineqconstraints]

    @no_annotations
    def reinit(self, x):
        if self.i % 10 == 0:
            Dx = 0.01
            x.assign(self.reinit_solver.solve(x, Dx))

    def eval_gradients(self, x):
        """Returns the triplet (gradJ(x), gradG(x), gradH(x))
        """
        self.i += 1

        dJ = self.dJ(x)
        if self.nconstraints == 0:
            dG = []
        else:
            dG = self.dG(x)
        if self.nineqconstraints == 0:
            dH = []
        else:
            dH = self.dH(x)
        # Regularize all gradients
        self.reg_solver.solve(self.gradJ, dJ)

        for gradHi, dHi in zip(self.gradH, dH):
            self.reg_solver(gradHi, dHi)
        for gradGi, dGi in zip(self.gradG, dG):
            self.reg_solver(gradGi, dGi)

        return (self.gradJ, self.gradG, self.gradH)

    def retract(self, x, dx):
        import numpy as np

        maxv = np.max(x.vector()[:])
        hmin = 0.02414
        dt = 0.1 * 1.0 * hmin / maxv
        # dt = 0.01
        self.newphi.assign(
            self.hj_solver.solve(Constant(-1.0) * dx, x, steps=1, dt=dt), annotate=False
        )
        return self.newphi

    def restore(self):
        self.reg_solver.update_beta_param(self.beta_param * 0.1)
        self.beta_param *= 0.1
        print(f"New regularization parameter: {self.beta_param}")

    @no_annotations
    def inner_product(self, x, y):
        return assemble(inner(x, y) * dx)

