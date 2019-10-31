from firedrake import (VectorElement, FiniteElement, FunctionSpace,
                        Constant, Function, TestFunction, split, as_vector,
                        DirichletBC, SpatialCoordinate, solve, FacetNormal,
                        adjoint, replace, CellDiameter, derivative, TrialFunction,
                        homogenize, assemble)

from ufl import (grad, inner, dx, div, exp, sin, pi, dot, ds, avg, jump, dS)

from parameters_heat_exch import (INMOUTH2, INMOUTH1, line_sep, dist_center, inlet_width,
                                WALLS, INLET1, INLET2, OUTLET1, OUTLET2, width)

from functools import partial

mu = Constant(1e-2)                   # viscosity
alphamax = 2.5 * mu / (1e-7)
alphamin = 1e-12
epsilon = Constant(1000.0)
parameters = {
        "mat_type" : "aij",
        "ksp_type" : "preonly",
        "pc_type" : "lu",
        "pc_factor_mat_solver_type" : "mumps"
        }
ks = Constant(1e0)
cp_value = 5.0e3
cp = Constant(cp_value)
u_inflow = 2e-1
Pe = u_inflow * width * cp_value / ks.values()[0]
tin1 = Constant(10.0)
tin2 = Constant(100.0)
print("Peclet number: {:.5f}".format(Pe))


def hs(phi, epsilon):
    return Constant(alphamax)*Constant(1.0) / ( Constant(1.0) + exp(-epsilon*phi)) + Constant(alphamin)

def ef1(u, v, p, q, phi):
    a_fluid = mu*inner(grad(u), grad(v)) + inner(grad(p), v) + q*div(u)
    darcy_term = inner(u, v)
    return a_fluid*dx + hs(-phi, epsilon)*darcy_term*dx(0) + alphamax*darcy_term*dx(INMOUTH2)

def ef2(u, v, p, q, phi):
    a_fluid = mu*inner(grad(u), grad(v)) + inner(grad(p), v) + q*div(u)
    darcy_term = inner(u, v)
    return a_fluid*dx + hs(phi, epsilon)*darcy_term*dx(0) + alphamax*darcy_term*dx(INMOUTH1)

def eTp(T, U1, U2, t, ks, cp, tin1, tin2, mesh):
    # TODO a dict for the DirichletBC, tin1 and ds marker
    w = TestFunction(T)

    # Mesh-related functions
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    u1, _ = split(U1)
    u2, _ = split(U2)

    def upwind(u):
        return (dot(u, n) + abs(dot(u, n)))/2.0
    u1n = upwind(u1)
    u2n = upwind(u2)

    # Penalty term
    alpha = Constant(50000.0) # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
    # Bilinear form
    a_int = dot(grad(w), ks*grad(t) - cp*(u1 + u2)*t)*dx

    a_fac = Constant(-1.0)*ks*dot(avg(grad(w)), jump(t, n))*dS \
        + Constant(-1.0)*ks*dot(jump(w, n), avg(grad(t)))*dS \
          + ks('+')*(alpha('+')/avg(h))*dot(jump(w, n), jump(t, n))*dS

    a_vel = dot(jump(w), cp*(u1n('+') + u2n('+'))*t('+') - \
            cp*(u1n('-') + u2n('-'))*t('-'))*dS + \
            dot(w, cp*(u1n + u2n)*t)*ds

    a_bnd = dot(w, cp*dot(u1 + u2, n)*t)*(ds(INLET1) + ds(INLET2)) \
            + w*t*(ds(INLET1) + ds(INLET2)) \
            - w*tin1*ds(INLET1) \
            - w*tin2*ds(INLET2) \
            + alpha/h * ks *w*t*(ds(INLET1) + ds(INLET2)) \
            - ks * dot(grad(w), t*n)*(ds(INLET1) + ds(INLET2)) \
            - ks * dot(grad(t), w*n)*(ds(INLET1) + ds(INLET2))

    aT = a_int + a_fac + a_vel + a_bnd

    LT_bnd = alpha/h * ks * tin1 * w * ds(INLET1)  \
            + alpha/h * ks * tin2 * w * ds(INLET2) \
            - tin1 * ks * dot(grad(w), n) * ds(INLET1) \
            - tin2 * ks * dot(grad(w), n) * ds(INLET2)
    eT = aT - LT_bnd

    return eT


class OptimizationProblem(object):

    """ Define the forward problem, the adjoint,
    the derivatives and the cost function
    This will be replaced once pyadjoint can
    calculate derivatives given a level set
    function.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        # Build function space
        P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        TH = P2*P1
        W = FunctionSpace(mesh, TH)
        self.W = W

        U = Function(W)
        self.U = U
        u, p = split(U)
        self.U1, self.U2 = Function(W), Function(W)
        V = TestFunction(W)
        v, q = split(V)

        self.stks1 = partial(ef1, u, v, p, q)
        self.stks2 = partial(ef2, u, v, p, q)

        # Dirichelt boundary conditions
        X = SpatialCoordinate(mesh)
        _, y = X[0], X[1]
        inflow1 = as_vector([u_inflow*sin(((y - (line_sep - (dist_center + inlet_width))) * pi )/ inlet_width), 0.0])
        inflow2 = as_vector([u_inflow*sin(((y - (line_sep + dist_center)) * pi )/ inlet_width), 0.0])

        noslip = Constant((0.0, 0.0))

        # Stokes 1
        bcs1_1 = DirichletBC(W.sub(0), noslip, WALLS)
        bcs1_2 = DirichletBC(W.sub(0), inflow1, INLET1)
        bcs1_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET1)
        bcs1_4 = DirichletBC(W.sub(0), noslip, INLET2)
        bcs1_5 = DirichletBC(W.sub(0), noslip, OUTLET2)
        self.bcs1 = [bcs1_1,bcs1_2,bcs1_3,bcs1_4, bcs1_5]

        # Stokes 2
        bcs2_1 = DirichletBC(W.sub(0), noslip, WALLS)
        bcs2_2 = DirichletBC(W.sub(0), inflow2, INLET2)
        bcs2_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET2)
        bcs2_4 = DirichletBC(W.sub(0), noslip, INLET1)
        bcs2_5 = DirichletBC(W.sub(0), noslip, OUTLET1)
        self.bcs2 = [bcs2_1,bcs2_2,bcs2_3,bcs2_4, bcs2_5]

        # Convection difussion equation
        T = FunctionSpace(mesh, 'DG', 1)
        self.t = Function(T, name="Temperature")
        self.T = T

    @staticmethod
    def cost_function(U1, U2, t, mesh):
        n = FacetNormal(mesh)
        u1, p1 = split(U1)
        _, p2 = split(U2)

        Power1 = Constant(8e3)*p1*ds(INLET1)
        Power2 = Constant(8e3)*p2*ds(INLET2)
        Jform = Constant(-1e5)*inner(t*u1, n)*ds(OUTLET1) + \
                Power1 + Power2
        return Jform

    def cost_function_evaluation(self, phi):

        solve(self.stks1(phi)==0, self.U, bcs=self.bcs1, solver_parameters=parameters)
        self.U1.assign(self.U)
        solve(self.stks2(phi)==0, self.U, bcs=self.bcs2, solver_parameters=parameters)
        self.U2.assign(self.U)
        eT = eTp(self.T, self.U1, self.U2, self.t, ks, cp, tin1, tin2, self.mesh)
        solve(eT==0, self.t, solver_parameters=parameters)

        return assemble(self.cost_function(self.U1, self.U2, self.t, self.mesh))

    def derivative_evaluation(self, phi):
        # Foward and adjoint problems problem
        m = Function(self.T)
        XSI, PSI = Function(self.W), Function(self.W)
        deltat = TrialFunction(self.T)
        w = TestFunction(self.T)
        V = TestFunction(self.W)

        Jform = self.cost_function(self.U1, self.U2, self.t, self.mesh)

        eT = eTp(self.T, self.U1, self.U2, self.t, ks, cp, tin1, tin2, self.mesh)
        solve(adjoint(derivative(eT, self.t, deltat))==-derivative(Jform, self.t, w), m,
                solver_parameters=parameters)

        solve(adjoint(derivative(self.stks1(phi), self.U)) == -derivative(replace(eT, {w: m}), self.U1, V)
                - derivative(Jform, self.U1, V), XSI, bcs=homogenize(self.bcs1), solver_parameters=parameters)

        solve(adjoint(derivative(self.stks2(phi), self.U)) == (-derivative(replace(eT, {w: m}), self.U2, V)
            - derivative(Jform, self.U2, V)), PSI, bcs=homogenize(self.bcs2), solver_parameters=parameters)


        Lagr = replace(eT, {w: m}) + replace(self.stks1(phi), {self.U: self.U1, V: XSI}) + replace(self.stks2(phi), {self.U: self.U2, V: PSI}) + Jform
        X = SpatialCoordinate(self.mesh)
        dL = derivative(Lagr, X)
        dJ = assemble(dL)

        return dJ
