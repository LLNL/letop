from contextlib import ExitStack
from itertools import chain

import firedrake as fd
from firedrake import inner, div, dx, ds, dS
import firedrake.utils as utils
import firedrake.dmhooks as dmhooks
from firedrake.bcs import DirichletBC
from firedrake.linear_solver import LinearSolver
from firedrake.petsc import OptionsManager, PETSc
from firedrake.ufl_expr import FacetNormal
from ufl import Max, as_vector, diag, outer, lhs, rhs
from pyadjoint.enlisting import Enlist

from .hj_context import _HJContext
from typing import Union, List, Callable, Tuple
from functools import lru_cache
from ufl import Form


def _make_reasons(reasons):
    return dict(
        [(getattr(reasons, r), r) for r in dir(reasons) if not r.startswith("_")]
    )


TSReasons = _make_reasons(PETSc.TS.ConvergedReason())


def check_ts_convergence(ts):
    r = ts.getConvergedReason()
    # TODO: submit PR to petsc4py to add the following reasons
    # TSFORWARD_DIVERGED_LINEAR_SOLVE = -3,
    # TSADJOINT_DIVERGED_LINEAR_SOLVE = -4
    if r == -3:
        raise fd.ConvergenceError(
            "TS solve failed to converge. Reason: TSFORWARD_DIVERGED_LINEAR_SOLVE"
        )
    if r == -4:
        raise fd.ConvergenceError(
            "TS solve failed to converge. Reason: TSADJOINT_DIVERGED_LINEAR_SOLVE"
        )
    reason = TSReasons[r]
    if r < 0:
        raise fd.ConvergenceError(
            f"TS solve failed to converge after {ts.getStepNumber()} iterations. Reason: {reason}"
        )


@lru_cache(1000)
def flux_function_space(V: fd.VectorFunctionSpace):
    return fd.VectorFunctionSpace(V.ufl_domain(), "DG", V.ufl_element().degree())


def clip(vector, op):
    from ufl import conditional

    return as_vector([conditional(op(vi, 0.0), 1.0, 0.0) for vi in vector])


def flux_exterior_form(
    phi_n: fd.Function,
    bcs: Union[DirichletBC, List[DirichletBC]] = None,
    reinit=False,
) -> Form:
    V = phi_n.function_space()
    mesh = V.ufl_domain()
    n = FacetNormal(mesh)
    VDG0 = flux_function_space(V)
    p = fd.TrialFunction(VDG0)
    v = fd.TestFunction(VDG0)

    from ufl import ge, lt

    F1 = (
        inner(p, v) * dx
        + phi_n * div(v) * dx
        - inner(
            diag(phi_n("-") * clip(n("+"), ge) + phi_n("+") * clip(n("-"), ge)),
            outer(v("+"), n("+")) + outer(v("-"), n("-")),
        )
        * dS
    )

    if reinit:
        F1 -= inner(v, n) * phi_n * ds
    else:
        F1 -= inner(v, diag(phi_n * clip(n, lt)) * n) * ds
        if bcs:
            for bc in bcs:
                func_bc = bc.function_arg
                sub_ids = bc.sub_domain
                for sub_id in Enlist(sub_ids):
                    F1 -= inner(
                        v, diag(phi_n * clip(n, lt) + func_bc * clip(n, ge)) * n
                    ) * ds(sub_id)

    return F1


def flux_interior_form(
    phi_n: fd.Function,
    bcs: Union[DirichletBC, List[DirichletBC]] = None,
    reinit=False,
) -> Form:
    V = phi_n.function_space()
    mesh = V.ufl_domain()
    n = FacetNormal(mesh)
    VDG0 = flux_function_space(V)
    p = fd.TrialFunction(VDG0)
    v = fd.TestFunction(VDG0)

    from ufl import ge, lt

    F2 = (
        inner(p, v) * dx
        + phi_n * div(v) * dx
        - inner(
            diag(phi_n("+") * clip(n("+"), ge) + phi_n("-") * clip(n("-"), ge)),
            outer(v("+"), n("+")) + outer(v("-"), n("-")),
        )
        * dS
    )
    if reinit:
        F2 -= inner(v, n) * phi_n * ds
    else:
        F2 -= inner(v, diag(phi_n * clip(n, ge)) * n) * ds
        if bcs:
            for bc in bcs:
                func_bc = bc.function_arg
                sub_ids = bc.sub_domain
                for sub_id in Enlist(sub_ids):
                    F2 -= inner(
                        v, diag(phi_n * clip(n, ge) + func_bc * clip(n, lt)) * n
                    ) * ds(sub_id)

    return F2


class HamiltonJacobiProblem(object):
    r"""Hamilton Jacobi equation for a given function H"""

    def __init__(
        self,
        V: fd.FunctionSpace,
        phi_n: fd.Function,
        H: Callable[[fd.Function], fd.Form],
        dHdp: Callable[[fd.Function], fd.Form],
        tspan: Tuple,
        bcs: Union[DirichletBC, List[DirichletBC]] = None,
        form_compiler_parameters: dict = None,
        reinit: bool = False,
    ):
        """
            Solver for the Hamilton Jacobi (HJ) PDE with a Local Discontinuous Galerkin method based on
            Jue Yan, Stanley Osher,
            A local discontinuous Galerkin method for directly solving Hamilton–Jacobi equations,
            Journal of Computational Physics,
            Volume 230, Issue 1,
            2011,
            Pages 232-244,


            The boundary conditions distinguishes the signed distance function (reinit) as a special case
            I do not know yet how to come up with a unifying formulation. Probably I need to understand better
            how the characterisctic functions propagate. There is some useful information in this paper:
            "A high-order and interface-preserving discontinuous Galerkin method for level-set reinitialization"
            Jiaqi Zhang; Pengtao Yue;
            https://www.sciencedirect.com/science/article/pii/S0021999118307733
            in page 20 which might be useful.

        Args:
            V (fd.FunctionSpace): [description]
            phi_n (fd.Function): [description]
            H (Callable[[fd.Function], fd.Form]): Flux function
            dHdp (Callable[[fd.Function], fd.Form]): Flux derivative
            tspan (Tuple): Time span
            bcs (Union[DirichletBC, List[DirichletBC]], optional): Dirichlet boundary conditions. Defaults to None.
            form_compiler_parameters (dict, optional):  Defaults to None.
            reinit (bool, optional): Is this a reinitialization solver?. Defaults to False.

        Raises:
            TypeError: [description]

        Returns:
            [type]: [description]
        """

        from firedrake import solving

        self.bcs = solving._extract_bcs(bcs)

        self.phi_n = phi_n
        self.tspan = tspan
        if not isinstance(self.phi_n, fd.Function):
            raise TypeError(
                "Provided solution is a '%s', not a Function"
                % type(self.phi_n).__name__
            )

        # Store form compiler parameters
        self.form_compiler_parameters = form_compiler_parameters

        VDG0 = flux_function_space(V)

        F1 = flux_exterior_form(phi_n, bcs=bcs, reinit=reinit)
        F2 = flux_interior_form(phi_n, bcs=bcs, reinit=reinit)

        self.p1 = fd.Function(VDG0)
        self.p2 = fd.Function(VDG0)

        self.L1 = rhs(F1)
        solver_bjacobi = {"pc_type": "bjacobi", "sub_pc_type": "ilu"}
        self.solver1 = LinearSolver(
            fd.assemble(lhs(F1), form_compiler_parameters=form_compiler_parameters),
            solver_parameters=solver_bjacobi,
        )

        self.L2 = rhs(F2)
        self.solver2 = LinearSolver(
            fd.assemble(lhs(F2), form_compiler_parameters=form_compiler_parameters),
            solver_parameters=solver_bjacobi,
        )

        phi, rho = fd.TrialFunction(V), fd.TestFunction(V)

        def alpha(p1, p2):
            return as_vector(
                [
                    Max(dHdp_p1_i, dHdp_p2_i)
                    for dHdp_p1_i, dHdp_p2_i in zip(dHdp(p1), dHdp(p2))
                ]
            )

        b = (
            phi * rho * dx
            + (
                H((self.p1 + self.p2) / fd.Constant(2.0))
                - fd.Constant(1.0 / 2.0)
                * inner(alpha(self.p1, self.p2), (self.p1 - self.p2))
            )
            * rho
            * dx
        )
        self.Lb = rhs(b)
        self.solver_b = LinearSolver(
            fd.assemble(lhs(b), form_compiler_parameters=form_compiler_parameters),
            solver_parameters=solver_bjacobi,
        )

    def dirichlet_bcs(self):
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    @utils.cached_property
    def dm(self):
        return self.phi_n.function_space().dm


class HamiltonJacobiSolver(OptionsManager):
    r"""Solves a :class:`HamiltonJacobiProblem`."""

    def __init__(self, problem, **kwargs):
        r"""
        :arg problem: A :class:`HamiltonJacobiProblem` to solve.
        :kwarg options_prefix: an optional prefix used to distinguish
               PETSc options.  If not provided a unique prefix will be
               created.  Use this option if you want to pass options
               to the solver from the command line in addition to
               through the ``solver_parameters`` dict.
        :kwarg monitor_callback: A user-defined function that will
               be used at every timestep to display the iteration's progress.
        :kwarg solver_parameters: PETSc options
        """
        assert isinstance(problem, HamiltonJacobiProblem)

        parameters = kwargs.get("solver_parameters")
        if "parameters" in kwargs:
            raise TypeError("Use solver_parameters, not parameters")
        options_prefix = kwargs.get("options_prefix")
        monitor_callback = kwargs.get("monitor_callback")

        super(HamiltonJacobiSolver, self).__init__(parameters, options_prefix)

        appctx = kwargs.get("appctx")

        ctx = _HJContext(
            problem,
            appctx=appctx,
            options_prefix=self.options_prefix,
        )

        self.ts = PETSc.TS().create(comm=problem.dm.comm)
        self.ts.setType(PETSc.TS.Type.RK)

        self._problem = problem

        self._ctx = ctx
        self._work = problem.phi_n.dof_dset.layout_vec.duplicate()
        self.ts.setDM(problem.dm)

        for monitor in Enlist(monitor_callback):
            self.ts.setMonitor(monitor)

        self.tspan = problem.tspan
        self.ts.setTime(problem.tspan[0])
        self.ts.setMaxTime(problem.tspan[1])
        self.ts.setEquationType(PETSc.TS.EquationType.EXPLICIT)
        self.set_default_parameter("ts_exact_final_time", "stepover")

        self._set_problem_eval_funcs(ctx, problem)

        # Set from options now. We need the
        # DM with an app context in place so that if the DM is active
        # on a subKSP the context is available.
        dm = self.ts.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx, save=False):
            self.set_from_options(self.ts)

        # Necessary to reset the problem as ts.solve() starts from the last time step used.
        self._dt = self.ts.getTimeStep()
        self._transfer_operators = {}

    def _set_problem_eval_funcs(self, ctx, problem):
        r"""
        :arg problem: A :class:`DAEProblem` to solve.
        :arg ctx: A :class:`_HJContext` that contains the residual evaluations
        """
        ctx.set_rhsfunction(self.ts)

    def solve(self, u0: fd.Function = None):
        r"""Solve the time-dependent variational problem."""

        # Necessary to reset the problem as ts.solve() starts from the last time step used.
        self.ts.setTimeStep(self._dt)
        self.ts.setTime(self.tspan[0])
        self.ts.setStepNumber(0)

        if u0:
            self._problem.phi_n.assign(u0)

        self._set_problem_eval_funcs(
            self._ctx,
            self._problem,
        )

        # Make sure appcontext is attached to the DM before we solve.
        dm = self.ts.getDM()
        work = self._work
        with self._problem.phi_n.dat.vec as u:
            u.copy(work)
            with ExitStack() as stack:
                # Ensure options database has full set of options (so monitors
                # work right)
                for ctx in chain(
                    (
                        self.inserted_options(),
                        dmhooks.add_hooks(dm, self, appctx=self._ctx),
                    ),
                    self._transfer_operators,
                ):
                    stack.enter_context(ctx)
                self.ts.solve(work)
            work.copy(u)
        check_ts_convergence(self.ts)
        return self._problem.phi_n
