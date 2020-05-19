from firedrake import Function, File
import numpy as np

from lestofire.optimization import HJStabSolver, SignedDistanceSolver
from lestofire.utils import petsc_print

parameters = {
        "mat_type" : "aij",
        "ksp_type" : "preonly",
        "pc_type" : "lu",
        "pc_factor_mat_solver_type" : "mumps"
        }

class SteepestDescent(object):

    """ Steepest Descent for the level set evolution """

    def __init__(self, lagrangian, reg_solver, options={}, pvd_output=False, parameters={}):
        """
        Initializes the Steepest Descent algorithm with the Hamilton-Jacobi (HJ)
        equation.

        options:
            - hj_stab: Stabilization parameter to solve the HJ equation
            - hmin: Mesh minimum cell size for the CFL condition.
            - dt_scale: Scaling of the time step
            - HJ steps: Number of steps taken for the HJ equation
            - max_iter: Max number of optimization iterations"""

        self.set_options(options)
        self.lagrangian = lagrangian
        self.reg_solver = reg_solver
        self.pvd_output = pvd_output
        self.options = options

    @classmethod
    def default_options(cls):

        default = {'hj_stab': 1.5,
                    'hmin' : 0.01,
                    'dt_scale' : 1.0,
                    'max_iter' : 20,
                    'n_reinit' : 5,
                    'n_hj_steps': 5}

        return default

    def set_options(self, user_options):
        # Update options with provided dictionary.
        if not isinstance(user_options, dict):
            raise TypeError("Options have to be set with a dictionary object.")
        if hasattr(self,  'options'):
            options = self.options
        else:
            options = self.default_options()
        for key, val in user_options.items():
            if key not in options:
                raise KeyError("'{}' not a valid setting for {}".format(key, self.__class__.__name__))
            # TODO: check also that the provided value is admissible.
            options[key] = val
        self.options = options

        return options

    def solve(self, phi, velocity, iterative=False, tolerance=1e-6, It0=0):

        hj_stab = self.options['hj_stab']
        hmin = self.options['hmin']
        dt_scale = self.options['dt_scale']
        n_hj_steps = self.options['n_hj_steps']
        n_reinit = self.options['n_reinit']
        max_iter = self.options['max_iter']



        PHI = phi.function_space()
        phi_old = Function(PHI)
        mesh = PHI.mesh()
        hj_solver = HJStabSolver(mesh, PHI, c2_param=hj_stab, iterative=iterative)
        reinit_solver = SignedDistanceSolver(mesh, PHI, dt=1e-7, iterative=iterative)
        ## Line search parameters
        alpha0_init,ls,ls_max,gamma,gamma2 = [0.5,0,8,0.1,0.01]
        alpha0 = alpha0_init
        alpha  = alpha0 # Line search step

        ## Stopping criterion parameters
        Nx = 100
        It = 0
        global_It = It0
        stop = False

        Jarr = np.zeros(max_iter)

        if self.pvd_output:
            self.pvd_output.write(phi)

        tolerance_criteria = 100
        rel_change_J = 0.0
        while It < max_iter and stop == False:

            J = self.lagrangian(phi)
            Jarr[It] = J

            # CFL condition
            maxv = np.max(phi.vector()[:])
            dt = dt_scale * alpha * hmin / maxv
            # ------- LINE SEARCH ------------------------------------------
            if It > 0 and Jarr[It] > Jarr[It-1] and ls < ls_max:
                ls   += 1
                alpha *= gamma
                dt = dt_scale * alpha * hmin / maxv
                phi.assign(phi_old, annotate=False)
                phi.assign(hj_solver.solve(velocity, phi, steps=n_hj_steps, dt=dt), annotate=False)
            else:
                if self.lagrangian.constraints[0]:
                    fvalue = self.lagrangian.cost_function_value()
                    petsc_print('It: {0} Obj: {1:.5f} f(x): {2:.5f} '.
                                format(global_It + It, Jarr[It], fvalue), end='')

                    constr_values = [self.lagrangian.constraint_value(i) for i in range(self.lagrangian.m)]
                    [petsc_print("g[{0}]: {1:.5f} ".format(i, value), end='') for i, value in enumerate(constr_values)]

                    petsc_print('Error: {0:.4E} Step size: {1:.2E} Step iters: {2}'.
                                format(abs(rel_change_J), alpha, ls), end='')
                else:
                    petsc_print('It: {0} Obj: {3:.5f} Error: {1:.5f} Step size: {2:.5E} Step iters: {4}'.
                                format(global_It + It, abs(rel_change_J), alpha, Jarr[It], ls), end='')
                if ls == ls_max:
                    petsc_print(' \u274C Failed line search')
                else:
                    petsc_print(' \u2705')
                #------------ STOPPING CRITERION ---------------------------
                tolerance_criteria = max(abs(Jarr[It-5:It]-Jarr[It-1])) if It > 20 else 1e2

                rel_change_J = abs(Jarr[It-2] - Jarr[It-1]) / Jarr[0] if It > 1 else 1e2
                #print("Relative change in J: {0:.4E}, stopping criteria steepest descent: {1:.4E}".format(rel_change_J, tolerance))
                if It > 2 and abs(rel_change_J) < tolerance: # and ls < ls_max:
                    stop = True

                #if It > 20 and tolerance_criteria < tolerance*Jarr[It-1]/Nx**2/10 and ls < ls_max:
                #    stop = True

                #print('Compliance            : %.2f' % )
                #print('Volume fraction       : %.2f' % (vol/(lx*ly)))
                # Decrease or increase line search step
                if ls == ls_max: alpha0 = max(alpha0 * gamma2, 0.1*alpha0_init)
                if ls == 0:      alpha0 = min(alpha0 / gamma2, 1)
                # Reset alpha and line search index
                ls,alpha,It = [0,alpha0, It+1]

                dJ = self.lagrangian.derivative()
                self.reg_solver.solve(velocity, dJ)

                phi_old.assign(phi, annotate=False)
                phi.assign(hj_solver.solve(velocity, phi, steps=n_hj_steps, dt=dt), annotate=False)
                if self.pvd_output:
                    self.pvd_output.write(phi)

                # Reinit the level set function every five iterations.
                if np.mod(It, n_reinit) == 0:
                    Dx = hmin
                    phi.assign(reinit_solver.solve(phi, Dx), annotate=False)

        return Jarr, It
