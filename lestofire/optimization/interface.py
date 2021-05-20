# Copyright 2018-2019 CNRS, Ecole Polytechnique and Safran.
#
# This file is part of nullspace_optimizer.
#
# nullspace_optimizer is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# nullspace_optimizer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# A copy of the GNU General Public License is included below.
# For further information, see <http://www.gnu.org/licenses/>.

from lestofire.physics.utils import max_mesh_dimension
import firedrake as fd
from firedrake import inner, dx
from pyadjoint import Control, no_annotations
from pyadjoint.enlisting import Enlist
from lestofire.levelset import (
    LevelSetFunctional,
    RegularizationSolver,
)
from lestofire.optimization import (
    ReinitializationSolver,
    HamiltonJacobiSolver,
    HamiltonJacobiProblem,
)
from lestofire.physics import calculate_max_vel
from pyop2.profiling import timed_function
from ufl.algebra import Abs
from firedrake import PETSc


class Constraint(object):
    @no_annotations
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
        # TODO: Any way to check that the value saved in self.scalar_control corresponds to `x`?
        """Evaluate the constraint as G(x) - Gval <= 0

        Args:
            x ([type]): [description]

        Returns:
            [float]: [Constraint value. Violation if it is > 0]
        """

        return self.scalar_control.tape_value() - self.constraint_value

    def derivative(self):
        return self.rfnl.derivative()


class InfDimProblem(object):
    def __init__(
        self,
        cost_function,
        reg_solver,
        eqconstraints=None,
        ineqconstraints=None,
        reinit_steps=10,
        reinit_distance=0.05,
        solver_parameters=None,
    ):
        """Problem interface for the null-space solver

        Args:
            cost_function ([type]): [description]
            reg_solver ([type]): [description]
            eqconstraints ([type], optional): [description]. Defaults to None.
            ineqconstraints ([type], optional): [description]. Defaults to None.
            reinit_steps (int, optional): [description]. Defaults to 10.
            reinit_distance (int, optional): The reinitialization solver is activated
                                                    after the level set is shifted reinit_distance * D,
                                                    where D is the max dimensions of a mesh
                                                    Defaults to 0.1
            solver_parameters ([type], optional): [description]. Defaults to None.

        Raises:
            TypeError: [description]
            TypeError: [description]
            TypeError: [description]

        Returns:
            [type]: [description]
        """
        self.reinit_steps = reinit_steps
        if not isinstance(reg_solver, RegularizationSolver):
            raise TypeError(
                f"Provided regularization solver '{type(reg_solver).__name__}', is not a RegularizationSolver"
            )
        self.reg_solver = reg_solver
        assert len(cost_function.controls) < 2, "Only one control for now"
        self.phi = cost_function.level_set[0]
        self.newphi = fd.Function(self.phi.function_space())
        self.V = self.phi.function_space()
        self.Vvec = cost_function.controls[0].control.function_space()
        self.delta_x = fd.Function(self.Vvec)
        self.max_distance = reinit_distance * max_mesh_dimension(self.V.ufl_domain())
        self.accum_distance = 0.0

        def H(p):
            return inner(self.delta_x, p)

        def dHdp(p):
            return Abs(self.delta_x)

        tspan = [0, 1.0]
        problem = HamiltonJacobiProblem(
            self.V,
            self.phi,
            H,
            dHdp,
            tspan,
            reinit=True,
        )
        hj_solver_parameters = {
            "ts_type": "rk",
            "ts_rk_type": "5dp",
            "ts_atol": 1e-8,
            "ts_rtol": 1e-8,
            "ts_dt": 1e-5,
            "ts_exact_final_time": "matchstep",
            "ts_max_time": tspan[1],
            "ts_event_monitor": None,
            "ts_max_steps": 1000,
            "ts_adapt_type": "dsp",
        }
        if solver_parameters:
            if solver_parameters.get("hj_solver"):
                hj_solver_parameters.update(solver_parameters["hj_solver"])

        self.hj_solver = HamiltonJacobiSolver(
            problem,
            solver_parameters=hj_solver_parameters,
        )

        def event(ts, t, X, fvalue):
            max_vel = calculate_max_vel(self.delta_x)
            fvalue[0] = (self.accum_distance + max_vel * t) - self.max_distance

        def postevent(ts, events, t, X, forward):
            with self.phi.dat.vec_wo as v:
                X.copy(v)
            self.phi.assign(self.reinit_solver.solve(self.phi, 0.5))
            with self.phi.dat.vec_wo as v:
                v.copy(X)
            self.max_distance += self.max_distance


        direction = [1]
        terminate = [False]

        self.hj_solver.ts.setEventHandler(direction, terminate, event, postevent)
        self.hj_solver.ts.setEventTolerances(1e-4, vtol=[1e-4])

        reinit_solver_parameters = {
            "ts_type": "rk",
            "ts_rk_type": "5dp",
            "ts_atol": 1e-7,
            "ts_rtol": 1e-7,
            "ts_dt": 1e-5,
            "ts_converged_reason": None,
            "ts_monitor": None,
            "ts_adapt_type": "dsp",
        }
        if solver_parameters:
            if solver_parameters.get("reinit_solver"):
                reinit_solver_parameters.update(solver_parameters["reinit_solver"])

        self.reinit_solver = ReinitializationSolver(
            self.V,
            solver_parameters=reinit_solver_parameters,
        )

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

        self.gradJ = fd.Function(self.Vvec)

        self.gradH = [fd.Function(self.Vvec) for _ in self.ineqconstraints]
        self.gradG = [fd.Function(self.Vvec) for _ in self.eqconstraints]

        self.cost_function = cost_function

        self.i = 0  # iteration count

        self.beta_param = reg_solver.beta_param.values()[0]

    def fespace(self):
        return self.Vvec

    def x0(self):
        return self.phi

    def eval(self, x):
        """Returns the triplet (J(x),G(x),H(x))"""
        return (self.J(x), self.G(x), self.H(x))

    @timed_function("Cost function")
    def J(self, x):
        return self.cost_function(x)

    @timed_function("Cost function gradient")
    def dJ(self, x):
        return self.cost_function.derivative()

    @timed_function("Equality constraint function")
    def G(self, x):
        return [eqconstr.evaluate(x) for eqconstr in self.eqconstraints]

    @timed_function("Equality constraint function gradient")
    def dG(self, x):
        return [eqconstr.derivative() for eqconstr in self.eqconstraints]

    @timed_function("Inequality constraint function")
    def H(self, x):
        return [ineqconstr.evaluate(x) for ineqconstr in self.ineqconstraints]

    @timed_function("Inequality constraint function gradient")
    def dH(self, x):
        return [ineqconstr.derivative() for ineqconstr in self.ineqconstraints]

    @no_annotations
    def reinit(self, x):
        pass

    @no_annotations
    def eval_gradients(self, x):
        """Returns the triplet (gradJ(x), gradG(x), gradH(x))"""
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

    @no_annotations
    def retract(self, phi_n, delta_x, scaling=1):
        self.hj_solver.ts.setMaxTime(scaling)
        phi_n = self.hj_solver.solve(phi_n)

        max_vel = calculate_max_vel(delta_x)
        self.accum_distance += max_vel * scaling

        conv = self.hj_solver.ts.getConvergedReason()
        if conv == 2:
            fd.warning(
                f"Maximum number of time steps (1000) reached. \
                        Consider making the optimization time step 'dt' shorter"
            )
        return phi_n

    def restore(self):
        pass

    @no_annotations
    def inner_product(self, x, y):
        return fd.assemble(inner(x, y) * dx)

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
