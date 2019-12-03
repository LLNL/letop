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

        self.opti_solver = SteepestDescent(lagrangian, reg_solver, options=options)


    def solve(self, phi, velocity, solver_parameters=parameters, tolerance=5e-3):
        it_max = 100
        it = 0
        stop_value = 1e-1
        while stop_value > 1e-6 and it < it_max:
            print(colored("Outer It.: {:d} ".format(it), 'green'))
            it = it + 1

            Jarr = self.opti_solver.solve(phi, velocity, solver_parameters, tolerance)
            tolerance *= 0.5

            stop_value = self.lagrangian.stop_criteria()
            self.lagrangian.update_augmented_lagrangian()
            tolerance *= 0.8
            print(colored("Stopping criteria {0:.5f}".format(stop_value), 'blue'))

        return Jarr
