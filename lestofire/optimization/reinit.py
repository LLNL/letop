from firedrake import FunctionSpace, TrialFunction, \
                    TestFunction, \
                    sqrt, Constant, \
                    dx, grad, inner, dot, \
                    Function, \
                    solve, assemble

default_solver_parameters = {
    "mat_type" : "aij",
    "ksp_type" : "preonly",
    "pc_type" : "lu",
    "pc_factor_mat_solver_type" : "mumps"
    }
class SignedDistanceSolver(object):
    def __init__(self, mesh, PHI, alpha=10.0, dt=1e-6, n_steps=10):
        self.PHI = PHI
        self.mesh = mesh
        self.alpha = alpha
        self.dt = dt
        self.n_steps = n_steps

    def solve(self, phi_n, Dx, solver_parameters=default_solver_parameters):
        # phi_n     - is the  level  set  field  which  comes  from of the  main  algorithm
        # mesh - mesh  description
        # Dx    - mesh  size in the x-direction

        # Space  definition
        # Set the  initial  value
        phi = TrialFunction(self.PHI)
        phi0 = TrialFunction(self.PHI)
        w = TestFunction(self.PHI)
        # Gradient  norm
        def mgrad(b):
            return(sqrt(b.dx (0)**2 + b.dx (1)**2))
        # Time  step
        k = Constant(self.dt)
        # Time  step  Python/FEniCS  syntax
        phi0 = phi_n
        # Initial  value
        eps = Constant (1.0/Dx)
        eps = Constant(Dx)
        # Interface  thickness
        alpha = Constant (0.0625/ Dx)
        # Numerical  diffusion  parameter
        signp = phi_n/sqrt(phi_n*phi_n + eps*eps*mgrad(phi_n)*mgrad(phi_n))
        # FEM  linearization
        a = (phi/k)*w*dx
        L = (phi0/k)*w*dx + signp*(1.0 - sqrt(dot(grad(phi0),grad(phi0))))*w*dx - alpha*inner(grad(phi0),grad(w))*dx
        # Boundary  condition
        bc = []
        # Flag  setup
        E_old = 1e10
        cont = 0
        phi = Function(self.PHI)

        from firedrake import File
        for n in range(self.n_steps):
            solve(a == L, phi , bc, solver_parameters=solver_parameters)
            # Euclidean  norm
            error = (((phi - phi0)/k)**2)*dx
            E = sqrt(abs(assemble(error)))
            print("error:", E)
            phi0.assign(phi)

            # Divergence  flag
            if (E_old < E ):
                fail = 1
                print("*Diverges_at_the_re -initialization_level*", cont)
                break
            cont  +=1
            E_old = E
        return  phi
