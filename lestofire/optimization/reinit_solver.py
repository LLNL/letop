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
        self,
        V,
        monitor_callback=None,
        poststep: bool = True,
        solver_parameters: dict = None,
    ) -> None:
        """Reinitialization solver

        Args:
            V ([type]): [description]
            h_factor (float): Used to approximate the signed distance function
            monitor_callback ([type], optional): [description]. Defaults to None.
            stopping_criteria (float, optional): Min value for the time derivative. Defaults to 0.1.
            poststep (bool, optional): Choose to add the postep (to check the stopping criteria). Defaults to True
            solver_parameters (dict, optional): [description]. Defaults to None.
        """
        if solver_parameters:
            stopping_criteria = solver_parameters.get(
                "stopping_criteria", 0.1
            )
            h_factor = solver_parameters.get(
                "h_factor", 2.0
            )
        self.h_factor = h_factor
        self.monitor_callback = Enlist(monitor_callback)
        self.error_prev = 1.0
        self.stopping_criteria = stopping_criteria
        self.solver_parameters = solver_parameters
        self.poststep = poststep

        pass

    @no_annotations
    def solve(self, phi0: fd.Function, total_t: float = 1.0):
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
            print(f"Residual: {time_deriv}")

            current_t = ts.getTime()
            current_dt = ts.getTimeStep()
            if time_deriv < self.stopping_criteria or current_t > current_dt * 20:
                ts.setConvergedReason(PETSc.TS.ConvergedReason.CONVERGED_USER)
            elif time_deriv > 1e5 and ts.getStepNumber() > 5:
                ts.setConvergedReason(PETSc.TS.ConvergedReason.DIVERGED_STEP_REJECTED)
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
            solver_parameters=self.solver_parameters,
            monitor_callback=self.monitor_callback,
        )
        if self.poststep:
            solver.ts.setPostStep(poststep)
        solver.solve()

        return phi0
