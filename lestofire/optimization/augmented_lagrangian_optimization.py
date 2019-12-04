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


    def solve(self, phi, velocity, solver_parameters=parameters, tolerance=5e-3):
        it_max = 100
        it = 0
        stop_value = 1e-1
        while stop_value > self.stopping_criteria and it < it_max:
            print(colored("Outer It.: {:d} ".format(it), 'green'))
            it = it + 1

            print(colored("Lagrange mult value: {0:.5f}, Penalty: {1:.5f}".format(self.lagrangian.lagrange_multiplier(0), self.lagrangian.penalty(0)), 'red'))
            Jarr = self.opti_solver.solve(phi, velocity, solver_parameters, tolerance)
            tolerance *= 0.5

            self.lagrangian.update_augmented_lagrangian()
            stop_value = self.lagrangian.stop_criteria()
            tolerance *= 0.8
            print(colored("Stopping criteria {0:.5f}".format(stop_value), 'blue'))

        return Jarr
