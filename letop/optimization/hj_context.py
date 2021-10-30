from firedrake import (
    assemble,
    dmhooks,
    function,
)
from firedrake.exceptions import ConvergenceError
from firedrake.petsc import PETSc
from firedrake.utils import cached_property


def _make_reasons(reasons):
    return dict(
        [
            (getattr(reasons, r), r)
            for r in dir(reasons)
            if not r.startswith("_")
        ]
    )


TSReasons = _make_reasons(PETSc.TS.ConvergedReason())


def check_ts_convergence(ts):
    r = ts.getConvergedReason()
    # TODO: submit PR to petsc4py to add the following reasons
    # TSFORWARD_DIVERGED_LINEAR_SOLVE = -3,
    # TSADJOINT_DIVERGED_LINEAR_SOLVE = -4
    if r == -3:
        raise ConvergenceError(
            "TS solve failed to converge. Reason: TSFORWARD_DIVERGED_LINEAR_SOLVE"
        )
    if r == -4:
        raise ConvergenceError(
            "TS solve failed to converge. Reason: TSADJOINT_DIVERGED_LINEAR_SOLVE"
        )
    reason = TSReasons[r]
    if r < 0:
        raise ConvergenceError(
            f"TS solve failed to converge after {ts.getStepNumber()} iterations. Reason: {reason}"
        )


class _HJContext(object):
    r"""
    Context holding information for TS callbacks.

    :arg problem: a :class:`HamiltonJacobiProlbem`.
    :arg appctx: Any extra information used in the assembler.  For the
        matrix-free case this will contain the Newton state in
        ``"state"``.
    :arg options_prefix: The options prefix of the TS.
    """

    def __init__(
        self,
        problem,
        appctx=None,
        options_prefix=None,
    ):
        from firedrake.bcs import DirichletBC

        self.options_prefix = options_prefix

        self._problem = problem

        self.fcp = problem.form_compiler_parameters

        if appctx is None:
            appctx = {}
        appctx.setdefault("state", self._problem.phi_n)
        appctx.setdefault("form_compiler_parameters", self.fcp)

        self.appctx = appctx

        self.bcs_F = [
            bc if isinstance(bc, DirichletBC) else bc._F for bc in problem.bcs
        ]

    def set_rhsfunction(self, ts):
        r"""Set the function to compute F(t,U,U_t) where F() = 0 is the DAE to be solved."""
        with self._F.dat.vec_wo as v:
            ts.setRHSFunction(self.form_function, f=v)

    @staticmethod
    def form_function(ts, t, X, F):
        r"""Form the residual for this problem

        :arg ts: a PETSc TS object
        :arg t: the time at step/stage being solved
        :arg X: state vector
        :arg F: function vector
        """
        dm = ts.getDM()
        ctx = dmhooks.get_appctx(dm)
        # X may not be the same vector as the vec behind self._problem.phi_n, so
        # copy guess in from X.
        with ctx._problem.phi_n.dat.vec_wo as v:
            X.copy(v)

        b1 = assemble(ctx._problem.L1)
        ctx._problem.solver1.solve(ctx._problem.p1, b1)

        b2 = assemble(ctx._problem.L2)
        ctx._problem.solver2.solve(ctx._problem.p2, b2)

        b3 = assemble(ctx._problem.Lb)
        ctx._problem.solver_b.solve(ctx._F, b3)

        # F may not be the same vector as self._F, so copy
        # residual out to F.
        with ctx._F.dat.vec_ro as v:
            v.copy(F)

    @cached_property
    def _F(self):
        return function.Function(self._problem.phi_n.function_space())
