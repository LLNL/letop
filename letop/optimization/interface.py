from letop.physics.utils import max_mesh_dimension
import firedrake as fd
from firedrake import inner, dx
from pyadjoint import Control, no_annotations
from pyadjoint.enlisting import Enlist
from letop.levelset import (
    LevelSetFunctional,
    RegularizationSolver,
)
from letop.optimization import (
    HamiltonJacobiCGSolver,
    ReinitSolverCG,
)
from letop.physics import calculate_max_vel, InteriorBC
from pyop2.profiling import timed_function
from ufl.algebra import Abs
from petsc4py.PETSc import TS



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
        reinit_distance=0.05,
        solver_parameters=None,
        output_dir=None,
    ):
        """Problem interface for the null-space solver

        Args:
            cost_function ([type]): [description]
            reg_solver ([type]): [description]
            eqconstraints ([type], optional): [description]. Defaults to None.
            ineqconstraints ([type], optional): [description]. Defaults to None.
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
        if not isinstance(reg_solver, RegularizationSolver):
            raise TypeError(
                f"Provided regularization solver '{type(reg_solver).__name__}',\
                  is not a RegularizationSolver"
            )
        self.reg_solver = reg_solver
        assert len(cost_function.controls) < 2, "Only one control for now"
        self.phi = cost_function.level_set[0]
        # Copy for the HamiltonJacobiCGSolver
        self.phi_hj = fd.Function(cost_function.level_set[0], name="hamilton-jacobi")
        # Copy for the Line search
        self.phi_ls = fd.Function(cost_function.level_set[0], name="line-search")
        # Copy for the Regularization solver
        self.phi_rs = fd.Function(cost_function.level_set[0], name='reinit')
        self.V = self.phi.function_space()
        self.Vvec = cost_function.controls[0].control.function_space()
        self.delta_x = fd.Function(self.Vvec)
        self.max_distance = reinit_distance * max_mesh_dimension(
            self.V.ufl_domain()
        )
        self.current_max_distance = self.max_distance
        self.current_max_distance_at_t0 = self.current_max_distance
        self.accum_distance = 0.0
        self.last_distance = 0.0
        self.output_dir = output_dir
        self.accept_iteration = False
        self.termination_event = None

        V_elem = self.V.ufl_element()
        if V_elem.family() in ["TensorProductElement", "Lagrange"]:
            if V_elem.family() == "TensorProductElement":
                assert (
                    V_elem.sub_elements()[0].family() == "Q"
                    and V_elem.sub_elements()[1].family() == "Lagrange"
                ), "Only Lagrange basis"
            self.build_cg_solvers(
                solver_parameters,
            )
        else:
            raise RuntimeError(
                f"Level set function element {self.V.ufl_element()} not supported."
            )

        def event(ts, t, X, fvalue):
            max_vel = calculate_max_vel(self.delta_x)
            fvalue[0] = (
                self.accum_distance + max_vel * t
            ) - self.current_max_distance

        def postevent(ts, events, t, X, forward):
            with self.phi_hj.dat.vec_wo as v:
                X.copy(v)
            self.phi_hj.assign(self.reinit_solver.solve(self.phi_hj))
            with self.phi_hj.dat.vec_wo as v:
                v.copy(X)
            self.current_max_distance += self.max_distance

        direction = [1]
        terminate = [False]

        self.hj_solver.ts.setEventHandler(
            direction, terminate, event, postevent
        )
        self.hj_solver.ts.setEventTolerances(1e-4, vtol=[1e-4])

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
                        f"Provided inequality constraint '{type(ineqconstr).__name__}',\
                          not a Constraint"
                    )
        else:
            self.ineqconstraints = []

        self.n_eqconstraints = len(self.eqconstraints)
        self.n_ineqconstraints = len(self.ineqconstraints)

        self.gradJ = fd.Function(self.Vvec)

        self.gradH = [fd.Function(self.Vvec) for _ in self.ineqconstraints]
        self.gradG = [fd.Function(self.Vvec) for _ in self.eqconstraints]

        self.cost_function = cost_function

        self.i = 0  # iteration count

        self.beta_param = reg_solver.beta_param.values()[0]


    def set_termination_event(
        self, termination_event, termination_tolerance=1e-2
    ):

        if termination_event and not isinstance(termination_event(), float):
            raise TypeError(f"termination_event must return a float")

        self.termination_event = termination_event
        self.termination_tolerance = termination_tolerance

    def build_cg_solvers(self, solver_parameters=None):
        hj_solver_parameters = None
        reinit_solver_parameters = None
        if solver_parameters:
            if solver_parameters.get("hj_solver"):
                hj_solver_parameters = solver_parameters["hj_solver"]
            if solver_parameters.get("reinit_solver"):
                reinit_solver_parameters = solver_parameters["reinit_solver"]

        self.hj_solver = HamiltonJacobiCGSolver(
            self.V,
            self.delta_x,
            self.phi_hj,
            solver_parameters=hj_solver_parameters,
        )
        self.reinit_solver = ReinitSolverCG(
            self.phi_rs, solver_parameters=reinit_solver_parameters
        )

    def fespace(self):
        return self.Vvec

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
        self.accum_distance += self.last_distance
        self.i += 1
        self.phi.assign(x)

        if self.termination_event:
            event_value = self.termination_event()
            if event_value < self.termination_tolerance:
                self.accept_iteration = True

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

    def reset_distance(self):
        """Necessary in cases where we have performed a reinitialization
        but the new level set was scraped by the line search
        """
        self.current_max_distance = self.current_max_distance_at_t0

    def velocity_scale(self, delta_x):
        return calculate_max_vel(delta_x)

    @no_annotations
    def retract(self, input_phi, delta_x, scaling=1):
        """input_phi is not modified,
            output_phi refers to problem.phi
        Args:
            input_phi ([type]): [description]
            delta_x ([type]): [description]
            scaling (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
        self.current_max_distance_at_t0 = self.current_max_distance
        self.hj_solver.ts.setMaxTime(scaling)
        self.hj_solver.ts.setTimeStep(1e-4)
        self.hj_solver.ts.setTime(0.0)
        self.hj_solver.ts.setStepNumber(0)
        try:
            self.phi_hj.assign(input_phi)
            self.hj_solver.solve()
        except:
            reason = self.hj_solver.ts.getConvergedReason()
            print(f"Time stepping of Hamilton Jacobi failed. Reason: {reason}")
            if self.output_dir:
                print(f"Printing last solution to {self.output_dir}")
                fd.File(f"{self.output_dir}/failed_hj.pvd").write(input_phi)

        max_vel = calculate_max_vel(delta_x)
        self.last_distance = max_vel * scaling

        conv = self.hj_solver.ts.getConvergedReason()
        rtol, atol = (
            self.hj_solver.parameters["ts_rtol"],
            self.hj_solver.parameters["ts_atol"],
        )
        max_steps = self.hj_solver.parameters.get("ts_max_steps", 800)
        current_time = self.hj_solver.ts.getTime()
        total_time = self.hj_solver.ts.getMaxTime()
        if conv == TS.ConvergedReason.CONVERGED_ITS:
            warning = (
                f"Maximum number of time steps {self.hj_solver.ts.getStepNumber()} reached."
                f"Current time is only: {current_time} for total time: {total_time}."
                "Consider making the optimization time step dt shorter."
                "Restarting this step with more time steps and tighter tolerances if applicable."
            )
            fd.warning(warning)

            # Tighten tolerances if we are really far
            if current_time / total_time < 0.2:
                current_rtol, current_atol = self.hj_solver.ts.getTolerances()
                new_rtol, new_atol = max(rtol / 200, current_rtol / 10), max(
                    atol / 200, current_atol / 10
                )
                self.hj_solver.ts.setTolerances(rtol=new_rtol, atol=new_atol)

            # Relax max time steps
            current_max_steps = self.hj_solver.ts.getMaxSteps()
            new_max_steps = min(current_max_steps * 1.5, max_steps * 3)
            self.hj_solver.ts.setMaxSteps(new_max_steps)

        elif conv == TS.ConvergedReason.CONVERGED_TIME:
            self.hj_solver.ts.setTolerances(rtol=rtol, atol=atol)

    def restore(self):
        pass

    @no_annotations
    def inner_product(self, x, y):
        return fd.assemble(inner(x, y) * dx)

    def accept(self):
        return self.accept_iteration
