from lestofire.levelset import (
    LevelSetFunctional,
    RegularizationSolver,
)
from lestofire.optimization import HJStabSolver, ReinitSolver
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
        if not isinstance(rfnl, LevelSetFunctional):
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

    def derivative(self):
        return self.rfnl.derivative()


class InfDimProblem(object):
    def __init__(
        self,
        cost_function,
        reg_solver,
        hj_solver,
        reinit_solver,
        eqconstraints=None,
        ineqconstraints=None,
    ):
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
        if not isinstance(reinit_solver, ReinitSolver):
            raise TypeError(
                f"Provided reinitialization solver '{type(hj_solver).__name__}', is not a ReinitSolver"
            )
        self.reinit_solver = reinit_solver

        if eqconstraints:
            self.eqconstraints = Enlist(eqconstraints)
            for constr in self.eqconstraints:
                if not isinstance(constr, Constraint):
                    raise TypeError(
                        f"Provided equality constraint '{type(constr).__name__}', not a Constraint"
                    )
        else:
            self.eqconstraints = []

        if ineqconstraints:
            self.ineqconstraints = Enlist(ineqconstraints)
            for ineqconstr in self.ineqconstraints:
                if not isinstance(ineqconstr, Constraint):
                    raise TypeError(
                        f"Provided inequality constraint '{type(ineqconstr).__name__}', not a Constraint"
                    )
        else:
            self.ineqconstraints = []

        assert len(cost_function.controls) < 2, "Only one control for now"
        self.V = cost_function.controls[0].control.function_space()

        self.gradJ = Function(self.V)

        self.gradH = [Function(self.V) for _ in self.ineqconstraints]
        self.gradG = [Function(self.V) for _ in self.eqconstraints]

        self.dx = Function(self.V)
        self.cost_function = cost_function

        self.phi = cost_function.level_set[0]
        self.newphi = Function(self.phi.function_space())
        self.i = 0  # iteration count

        self.h_size = 1e-7
        self.beta_param = reg_solver.beta_param.values()[0]

    def fespace(self):
        return self.V

    def x0(self):
        return self.phi

    def eval(self, x):
        """Returns the triplet (J(x),G(x),H(x))"""
        return (self.J(x), self.G(x), self.H(x))

    def J(self, x):
        return self.cost_function(x)

    def dJ(self, x):
        return self.cost_function.derivative()

    def G(self, x):
        return [eqconstr.evaluate(x) for eqconstr in self.eqconstraints]

    def dG(self, x):
        return [eqconstr.derivative() for eqconstr in self.eqconstraints]

    def H(self, x):
        return [ineqconstr.evaluate(x) for ineqconstr in self.ineqconstraints]

    def dH(self, x):
        return [ineqconstr.derivative() for ineqconstr in self.ineqconstraints]

    @no_annotations
    def reinit(self, x):
        if self.i % 10 == 0:
            Dx = 0.01
            x.assign(self.reinit_solver.solve(x, Dx))

    @no_annotations
    def eval_gradients(self, x):
        """Returns the triplet (gradJ(x), gradG(x), gradH(x))
        """
        self.i += 1
        self.phi.assign(x)

        dJ = self.dJ(x)
        dG = self.dG(x)
        dH = self.dH(x)

        # Regularize all gradients
        self.reg_solver.solve(self.gradJ, dJ)

        for gradHi, dHi in zip(self.gradH, dH):
            self.reg_solver.solve(gradHi, dHi)
        for gradGi, dGi in zip(self.gradG, dG):
            self.reg_solver.solve(gradGi, dGi)

        return (self.gradJ, self.gradG, self.gradH)

    def retract(self, x, dx):
        import numpy as np

        maxv = np.max(x.vector()[:])
        hmin = 0.02414
        dt = 0.1 * 1.0 * hmin / maxv
        # dt = 0.01
        self.phi.assign(
            self.hj_solver.solve(Constant(-1.0) * dx, x, steps=1, dt=dt), annotate=False
        )
        return self.phi

    def restore(self):
        self.reg_solver.update_beta_param(self.beta_param * 0.1)
        self.beta_param *= 0.1
        print(f"New regularization parameter: {self.beta_param}")

    @no_annotations
    def inner_product(self, x, y):
        return assemble(inner(x, y) * dx)

    def accept(self, results):
        """
        This function is called by nlspace_solve:
            - at the initialization
            - every time a new guess x is accepted on the optimization
              trajectory
        This allows to perform some post processing operations along the
        optimization trajectory. The function does not return any output but
        may update the dictionary `results` which may affect the optimization.
        Notably, the current point is stored in
            results['x'][-1]
        and an update of its value will be taken into account by nlspace_solve.

        Inputs:
            `results` : the current dictionary of results (see the function
                nlspace_solve)
        """
        pass
