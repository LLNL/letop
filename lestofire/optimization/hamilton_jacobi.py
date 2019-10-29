from firedrake import FunctionSpace, TrialFunction,  \
                    TestFunction, Constant, \
                    Function, \
                    FacetNormal, CellDiameter, \
                    grad, jump, avg, inner, \
                    dx, dS, solve, lhs, rhs, dot, ds, \
                    div


default_solver_parameters = {
    "mat_type" : "aij",
    "ksp_type" : "preonly",
    "pc_type" : "lu",
    "pc_factor_mat_solver_type" : "mumps"
    }
class HJStabSolver(object):
    def __init__(self, mesh, PHI, c2_param=0.05, f=Constant(0.0)):
        self.PHI = PHI
        self.mesh = mesh
        self.c2 = c2_param
        self.f = f

    def solve(self, beta, phi_n, steps=5, t=0, dt=1.0, bc=None, solver_parameters=default_solver_parameters):
        phi = TrialFunction(self.PHI)
        psi = TestFunction(self.PHI)
        n = FacetNormal(self.mesh)
        h = CellDiameter(self.mesh)
        self.f.t = t

        for i in range(steps):
            F = (phi - phi_n)/Constant(dt) * psi * dx \
                    + inner(beta, 1.0/2.0 * grad(phi + phi_n)) * psi * dx \
                + self.c2 * avg(h) * avg(h) * inner(jump(1.0/2.0*grad(phi + phi_n), n), jump(grad(psi), n)) * dS \
                - self.f * psi * dx
            phi_new = Function(self.PHI)
            if bc is None:
                solve(lhs(F)==rhs(F), phi_new, solver_parameters=solver_parameters)
            else:
                solve(lhs(F)==rhs(F), phi_new, bcs=bc, solver_parameters=solver_parameters)
            phi_n.assign(phi_new)

        return phi_n

class HJSUPG(object):
    def __init__(self, mesh, PHI, f=Constant(0.0)):
        self.PHI = PHI
        self.mesh = mesh
        self.f = f


    def solve(self, beta, phi_n, steps=5, t=0, dt=1.0, bc=None,):
        phi = TrialFunction(self.PHI)
        psi = TestFunction(self.PHI)
        n = FacetNormal(self.mesh)
        h = CellDiameter(self.mesh)
        self.f.t = t


        for i in range(steps):
            F = (phi - phi_n)/Constant(dt) * psi * dx \
                    + inner(beta, 1.0/2.0 * grad(phi + phi_n)) * psi * dx \
                - self.f * psi * dx

            # Residual
            r = ((1/Constant(dt))*(phi - phi_n) + 1.0/2.0*inner(beta, grad(phi)) + (1-1.0/2.0)*inner(beta,grad(phi_n)) - self.f) * inner(beta,grad(psi))


            # Add SUPG stabilisation terms
            from ufl import sqrt
            vnorm = sqrt(dot(beta, beta))
            tau = 0.5*h/(2.0*vnorm)
            tau = 2.0 * 1.0 /(sqrt((1.0 / Constant(dt*dt) + vnorm / h)))
            F += tau*r*dx

            phi_new = Function(self.PHI)

            if bc is None:
                solve(lhs(F)==rhs(F), phi_new)
            else:
                solve(lhs(F)==rhs(F), phi_new, bcs=bc)
            phi_n.assign(phi_new)

        return phi_n

class HJDG(object):
    def __init__(self, mesh, PHI, f=Constant(0.0)):
        self.PHI = PHI
        self.mesh = mesh
        self.f = f


    def solve(self, beta, phi_n, steps=5, t=0, dt=1.0, bc=None,):
        phi = TrialFunction(self.PHI)
        psi = TestFunction(self.PHI)
        n = FacetNormal(self.mesh)
        h = CellDiameter(self.mesh)
        self.f.t = t

        betan = dot(beta, n) + abs(dot(beta, n))/2.0

        for i in range(steps):
            F_int = (phi - phi_n)/Constant(dt) * psi * dx \
                    - inner(beta * 1.0/2.0 * (phi + phi_n), grad(psi)) * dx \
                - self.f * psi * dx
            F_face = dot(jump(psi), betan('+')*1.0/2.0*(phi + phi_n)('+') - betan('-')*1.0/2.0*(phi + phi_n)('-') )*dS + dot(psi, betan*1.0/2.0*(phi + phi_n))*ds

            F = F_int + F_face

            phi_new = Function(self.PHI)

            if bc is None:
                solve(lhs(F)==rhs(F), phi_new)
            else:
                solve(lhs(F)==rhs(F), phi_new, bcs=bc)
            phi_n.assign(phi_new)

        return phi_n
