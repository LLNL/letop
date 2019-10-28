from firedrake import (Mesh, FunctionSpace, Function, Constant,
                        SpatialCoordinate, VectorElement, FiniteElement,
                        TestFunction, TrialFunction, interpolate, File, DirichletBC, solve,
                        FacetNormal, CellDiameter)

from ufl import (sin, pi, split, inner, grad, div, exp,
                dx, as_vector, derivative, dot, avg, jump, ds, dS)


from parameters_heat_exch import height, width, inlet_width, dist_center, inlet_depth, shift_center, line_sep
from parameters_heat_exch import MOUTH1, MOUTH2, INLET1, INLET2, OUTLET1, OUTLET2, WALLS

def main():

    mesh = Mesh('./mesh_heat_exchanger.msh')
    x, y = SpatialCoordinate(mesh)

    phi_expr = -y + line_sep \
                        + (y > line_sep + 0.2)*(-2.0*sin((y-line_sep - 0.2)*pi/0.5))*sin(x*pi/width) \
                        - (y < line_sep)*(0.5*sin((y + line_sep/3.0)*pi/(2.0*line_sep/3.0)))* \
                                sin(x*pi*2.0/width)

    PHI = FunctionSpace(mesh, 'CG', 1)
    phi = interpolate(phi_expr , PHI)
    File("phi_initial.pvd").write(phi)

    # Build function space
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2*P1
    W = FunctionSpace(mesh, TH)

    U = Function(W)
    u, p = split(U)
    U1, U2 = Function(W), Function(W)
    V = TestFunction(W)
    v, q = split(V)

    f = Constant((0.0, 0.0))
    mu = Constant(1e-2)                   # viscosity
    a_fluid = mu*inner(grad(u), grad(v)) + inner(grad(p), v) + q*div(u)
    darcy_term = inner(u, v)

    alphamax = 1e4
    alphamin = 1e-9
    epsilon = Constant(100.0)

    def hs(phi, epsilon):
        return Constant(alphamax)*Constant(1.0) / ( Constant(1.0) + exp(-epsilon*phi)) + Constant(alphamin)

    def e1f(phi):
        return a_fluid*(dx(0) + dx(MOUTH1) + dx(MOUTH2)) + hs(-phi, epsilon)*darcy_term*dx(0) + alphamax*darcy_term*dx(MOUTH2)
    def e2f(phi):
        return a_fluid*(dx(0) + dx(MOUTH1) + dx(MOUTH2)) + hs(phi, epsilon)*darcy_term*dx(0) + alphamax*darcy_term*dx(MOUTH1)

    u_inflow = 2e-1
    inflow1 = as_vector([u_inflow*sin(((y - (line_sep - (dist_center + inlet_width))) * pi )/ inlet_width), 0.0])
    inflow2 = as_vector([u_inflow*sin(((y - (line_sep + dist_center)) * pi )/ inlet_width), 0.0])

    pressure_outlet = Constant(0.0)
    noslip = Constant((0.0, 0.0))

    # Stokes 1
    bcs1_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs1_2 = DirichletBC(W.sub(0), inflow1, INLET1)
    bcs1_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET1)
    bcs1_4 = DirichletBC(W.sub(0), noslip, INLET2)
    bcs1_5 = DirichletBC(W.sub(0), noslip, OUTLET2)

    # Stokes 2
    bcs2_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs2_2 = DirichletBC(W.sub(0), inflow2, INLET2)
    bcs2_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET2)
    bcs2_4 = DirichletBC(W.sub(0), noslip, INLET1)
    bcs2_5 = DirichletBC(W.sub(0), noslip, OUTLET1)

    # Stokes 1 problem
    parameters = {
            "mat_type" : "aij",
            "ksp_type" : "preonly",
            "pc_type" : "lu",
            "pc_factor_mat_solver_type" : "mumps"
            }

    solve(e1f(phi)==0, U, bcs=[bcs1_1,bcs1_2,bcs1_3,bcs1_4, bcs1_5], solver_parameters=parameters)
    U1.assign(U)
    solve(e2f(phi)==0, U, bcs=[bcs2_1,bcs2_2,bcs2_3,bcs2_4, bcs2_5], solver_parameters=parameters)
    U2.assign(U)

    u, p = U1.split()
    u.rename("Velocity")
    p.rename("Pressure")
    File("u1.pvd").write(u, p)
    u, p = U2.split()
    u.rename("Velocity")
    p.rename("Pressure")
    File("u2.pvd").write(u, p)


    # Convection difussion equation
    T = FunctionSpace(mesh, 'DG', 1)

    t = Function(T, name="Temperature")
    w = TestFunction(T)
    theta = TrialFunction(T)

    # Mesh-related functions
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    h_avg = (h('+') + h('-'))/2

    kf = Constant(1e0)
    ks = Constant(1e0)
    cp_value = 5.0e3
    cp = Constant(cp_value)

    Pe = u_inflow * width * cp_value / ks.values()[0]
    print("Peclet number: {:.5f}".format(Pe))

    # Temperature problem
    tin1 = Constant(10.0)
    tin2 = Constant(100.0)

    u1, p1 = split(U1)
    u2, p2 = split(U2)
    u1n = (dot(u1, n) + abs(dot(u1, n)))/2.0
    u2n = (dot(u2, n) + abs(dot(u2, n)))/2.0
    # Penalty term
    alpha = Constant(50000.0) # 5.0 worked really well where there was no convection. For larger Peclet number, larger alphas
    # Bilinear form
    a_int = dot(grad(w), ks*grad(t) - cp*(u1 + u2)*t)*dx

    a_fac = - ks*dot(avg(grad(w)), jump(t, n))*dS \
          - ks*dot(jump(w, n), avg(grad(t)))*dS \
          + ks('+')*(alpha('+')/avg(h))*dot(jump(w, n), jump(t, n))*dS

    a_vel = dot(jump(w), Constant(cp_value)*(u1n('+') + u2n('+'))*t('+') - \
            Constant(cp_value)*(u1n('-') + u2n('-'))*t('-'))*dS + \
            dot(w, Constant(cp_value)*(u1n + u2n)*t)*ds

    a_bnd = dot(w, Constant(cp_value)*dot(u1 + u2, n)*t)*(ds(INLET1) + ds(INLET2)) \
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

    solve(eT==0, t, solver_parameters=parameters)
    File("t.pvd").write(t)






if __name__ == '__main__':
    main()
