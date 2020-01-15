from lestofire.optimization import SteepestDescent

from termcolor import colored


parameters = {
        "mat_type" : "aij",
        "ksp_type" : "preonly",
        "pc_type" : "lu",
        "pc_factor_mat_solver_type" : "mumps"
        }

class AugmentedLagrangianOptimization(object):

    """Implementes the Augmented Lagrangian Algorithm for constrained
        problems. It only works if the LevelSetLagrangian contains a constraint
        and it is set up as Augmented Lagrangian
        """

    def __init__(self, lagrangian, reg_solver, options={}, pvd_output=False, parameters={}):
        """
        Initializes the Augmented Lagriangian algorithm with the Steepest Descent
        algorithm
        """

        self.lagrangian = lagrangian
        self.reg_solver = reg_solver
        self.pvd_output = pvd_output
        self.options = options

        if 'stopping_criteria' in options:
            self.stopping_criteria = options.pop('stopping_criteria')
        else:
            self.stopping_criteria = 1e-4


        self.opti_solver = SteepestDescent(lagrangian, reg_solver, options=options)


    def solve(self, phi, velocity, iterative=False, tolerance=5e-3):
        it_max = 100
        it = 0
        stop_value = 1e-1
        while stop_value > self.stopping_criteria and it < it_max:
            print(colored("Outer It.: {:d} ".format(it), 'green'), flush=True)
            it = it + 1

            m_constr = range(self.lagrangian.m)
            lagr_mults = [self.lagrangian.lagrange_multiplier(i) for i in m_constr]
            c_penalties = [self.lagrangian.penalty(i) for i in m_constr]
            [print(colored("Lagr[{0}]: {1:.5f}. c[{0}]: {2:.5f}".format(i, lagr, penalty), 'green'), flush=True)
                    for i, (lagr, penalty) in enumerate(zip(lagr_mults, c_penalties)) ]
            Jarr = self.opti_solver.solve(phi, velocity, iterative=iterative, tolerance=tolerance)

            if tolerance > 1e-5:
                tolerance *= 0.5*0.8

            self.lagrangian.update_augmented_lagrangian()
            stop_value = self.lagrangian.stop_criteria()
            print(colored("Stopping criteria {0:.5f}".format(stop_value), 'blue'), flush=True)

        return Jarr
