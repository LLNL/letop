from firedrake import Function, File
import numpy as np

import sys
sys.path.append('..')
from lestofire import HJStabSolver, SignedDistanceSolver

parameters = {
        "mat_type" : "aij",
        "ksp_type" : "preonly",
        "pc_type" : "lu",
        "pc_factor_mat_solver_type" : "mumps"
        }

class SteepestDescent(object):

    """ Steepest Descent for the level set evolution """

    def __init__(self, lagrangian, reg_solver, hmin=0.0094, pvd_output=False, parameters={}):
        """TODO: to be defined. """

        self.lagrangian = lagrangian
        self.reg_solver = reg_solver
        self.pvd_output = pvd_output
        self.hmin = hmin

    def solve(self, phi, velocity, solver_parameters=parameters):
        PHI = phi.function_space()
        phi_old = Function(PHI)
        mesh = PHI.mesh()
        hj_solver = HJStabSolver(mesh, PHI, c2_param=1.5)
        reinit_solver = SignedDistanceSolver(mesh, PHI, dt=1e-6)
        ## Line search parameters
        alpha0_init,ls,ls_max,gamma,gamma2 = [0.5,0,8,0.1,0.1]
        alpha0 = alpha0_init
        alpha  = alpha0 # Line search step

        ## Stopping criterion parameters
        Nx = 100
        ItMax,It,stop = [int(1.5*Nx), 0, False]

        Jarr = np.zeros( ItMax )

        if self.pvd_output:
            self.pvd_output.write(phi)

        while It < ItMax and stop == False:

            J = self.lagrangian(phi)
            Jarr[It] = J

            # CFL condition
            maxv = np.max(phi.vector()[:])
            dt = 0.1 * alpha * self.hmin / maxv
            print("dt: {:.5f}".format(dt))
            # ------- LINE SEARCH ------------------------------------------
            if It > 0 and Jarr[It] > Jarr[It-1] and ls < ls_max:
                ls   += 1
                alpha *= gamma
                phi.assign(phi_old)
                phi.assign(hj_solver.solve(velocity, phi, steps=3, dt=dt))
                print('Line search iteration : %s' % ls)
                print('Line search step : %.8f' % alpha)
                print('Function value        : %.10f' % Jarr[It])
            else:
                print('************ ITERATION NUMBER %s' % It)
                print('Function value        : %.10f' % Jarr[It])
                #print('Compliance            : %.2f' % )
                #print('Volume fraction       : %.2f' % (vol/(lx*ly)))
                # Decrease or increase line search step
                if ls == ls_max: alpha0 = max(alpha0 * gamma2, 0.1*alpha0_init)
                if ls == 0:      alpha0 = min(alpha0 / gamma2, 1)
                # Reset alpha and line search index
                ls,alpha,It = [0,alpha0, It+1]

                dJ = self.lagrangian.derivative()
                self.reg_solver.solve(velocity, dJ, solver_parameters=solver_parameters)

                phi_old.assign(phi)
                phi.assign(hj_solver.solve(velocity, phi, steps=3, dt=dt, solver_parameters=solver_parameters))
                if self.pvd_output:
                    self.pvd_output.write(phi)

                # Reinit the level set function every five iterations.
                if np.mod(It,5) == 0:
                    Dx = self.hmin
                    phi.assign(reinit_solver.solve(phi, Dx))
                #------------ STOPPING CRITERION ---------------------------
                if It>20 and max(abs(Jarr[It-5:It]-Jarr[It-1]))<2.0e-8*Jarr[It-1]/Nx**2/10:
                    stop = True
