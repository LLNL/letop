from .hj_solver import (
    flux_exterior_form,
    flux_interior_form,
    HamiltonJacobiProblem,
    HamiltonJacobiSolver,
    flux_function_space,
)
import firedrake as fd
from firedrake import inner, sqrt, lhs, rhs
from ufl.algebra import Abs
from pyadjoint import no_annotations
from pyadjoint.enlisting import Enlist
from firedrake import PETSc


class ReinitializationSolver(object):
    def __init__(
        self, V, h_factor, monitor_callback=None, stopping_criteria=0.1
    ) -> None:
        self.h_factor = h_factor
        self.monitor_callback = Enlist(monitor_callback)
        self.error_prev = 1.0
        self.stopping_criteria = stopping_criteria

        pass

    @no_annotations
    def solve(self, phi0: fd.Function, total_t: float, solver_parameters: dict = None):
        V = phi0.function_space()
        mesh = V.ufl_domain()
        F1 = flux_exterior_form(phi0, reinit=True)
        F2 = flux_interior_form(phi0, reinit=True)
        Delta_x = fd.CellDiameter(mesh) * self.h_factor

        def sign(phi, grad_phi):
            return phi / sqrt(
                phi * phi + Delta_x * Delta_x * (inner(grad_phi, grad_phi))
            )

        phi_signed = fd.Function(V)
        phi_signed.assign(phi0)
        phi_prev = fd.Function(V)
        phi_current = fd.Function(V)

        def poststep(ts):
            x = ts.getSolution()
            with phi_current.dat.vec as v:
                x.copy(v)
            self.error_prev = fd.errornorm(phi_prev, phi_current)
            time_deriv = self.error_prev / ts.getTimeStep()

            if time_deriv < self.stopping_criteria:
                ts.setConvergedReason(PETSc.TS.ConvergedReason.CONVERGED_USER)
            phi_prev.assign(phi_current)

        VDG0 = flux_function_space(V)
        p1_0 = fd.Function(VDG0)
        p2_0 = fd.Function(VDG0)
        L1 = rhs(F1)
        solver_bjacobi = {"pc_type": "bjacobi", "sub_pc_type": "ilu"}
        solver1 = fd.LinearSolver(
            fd.assemble(lhs(F1)), solver_parameters=solver_bjacobi
        )
        solver1.solve(p1_0, fd.assemble(L1))
        L2 = rhs(F2)
        solver_bjacobi = {"pc_type": "bjacobi", "sub_pc_type": "ilu"}
        solver2 = fd.LinearSolver(
            fd.assemble(lhs(F2)), solver_parameters=solver_bjacobi
        )
        solver2.solve(p2_0, fd.assemble(L2))

        grad_phi_0 = (p1_0 + p2_0) / fd.Constant(2.0)

        def H(p):
            return sign(phi_signed, grad_phi_0) * (sqrt(inner(p, p)) - 1)

        def dHdp(p):
            return Abs(sign(phi_signed, grad_phi_0) * p / sqrt(inner(p, p) + 1e-7))

        tspan = [0, total_t]
        problem = HamiltonJacobiProblem(
            V,
            phi0,
            H,
            dHdp,
            tspan,
            reinit=True,
        )
        solver = HamiltonJacobiSolver(
            problem,
            solver_parameters=solver_parameters,
            monitor_callback=self.monitor_callback,
        )
        # self.monitor_callback.append(monitor)
        solver.ts.setPostStep(poststep)
        solver.solve()

        return phi0
