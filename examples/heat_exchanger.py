from firedrake import (Mesh, FunctionSpace, Function, Constant,
                        SpatialCoordinate, VectorElement, FiniteElement,
                        TestFunction, TrialFunction, interpolate, File, DirichletBC, solve,
                        FacetNormal, CellSize, CellDiameter, homogenize, assemble, adjoint, derivative, replace)

from ufl import (sin, pi, split, inner, grad, div, exp,
                dx, as_vector,  dot, avg, jump, ds, dS)


from parameters_heat_exch import height, width, inlet_width, dist_center, inlet_depth, shift_center, line_sep
from parameters_heat_exch import MOUTH1, MOUTH2, INLET1, INLET2, OUTLET1, OUTLET2, WALLS

import sys
sys.path.append('..')
from lestofire import HJStabSolver, SignedDistanceSolver

def main():

    mesh = Mesh('./mesh_heat_exchanger.msh')
    X = SpatialCoordinate(mesh)
    x, y = X[0], X[1]

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

    noslip = Constant((0.0, 0.0))

    # Stokes 1
    bcs1_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs1_2 = DirichletBC(W.sub(0), inflow1, INLET1)
    bcs1_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET1)
    bcs1_4 = DirichletBC(W.sub(0), noslip, INLET2)
    bcs1_5 = DirichletBC(W.sub(0), noslip, OUTLET2)
    bcs1 = [bcs1_1,bcs1_2,bcs1_3,bcs1_4, bcs1_5]

    # Stokes 2
    bcs2_1 = DirichletBC(W.sub(0), noslip, WALLS)
    bcs2_2 = DirichletBC(W.sub(0), inflow2, INLET2)
    bcs2_3 = DirichletBC(W.sub(1), Constant(0.0), OUTLET2)
    bcs2_4 = DirichletBC(W.sub(0), noslip, INLET1)
    bcs2_5 = DirichletBC(W.sub(0), noslip, OUTLET1)
    bcs2 = [bcs2_1,bcs2_2,bcs2_3,bcs2_4, bcs2_5]

    # Stokes 1 problem
    parameters = {
            "mat_type" : "aij",
            "ksp_type" : "preonly",
            "pc_type" : "lu",
            "pc_factor_mat_solver_type" : "mumps"
            }

    solve(e1f(phi)==0, U, bcs=bcs1, solver_parameters=parameters)
    U1.assign(U)
    solve(e2f(phi)==0, U, bcs=bcs2, solver_parameters=parameters)
    U2.assign(U)

    u, p = U1.split()
    u.rename("Velocity")
    p.rename("Pressure")
    u, p = U2.split()
    u.rename("Velocity")
    p.rename("Pressure")


    # Convection difussion equation
    T = FunctionSpace(mesh, 'DG', 1)

    t = Function(T, name="Temperature")
    w = TestFunction(T)

    # Mesh-related functions
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)

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

    a_fac = Constant(-1.0)*ks*dot(avg(grad(w)), jump(t, n))*dS \
        + Constant(-1.0)*ks*dot(jump(w, n), avg(grad(t)))*dS \
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


    #### OPTIMIZATION
    Power1 = Constant(8e3)*p1*ds(INLET1)
    Power2 = Constant(8e3)*p2*ds(INLET2)
    Jform = Constant(-1e5)*inner(t*u1, n)*ds(OUTLET1) + \
            Power1 + Power2

    # Foward and adjoint problems problem
    m = Function(T)
    XSI, PSI = Function(W), Function(W)
    deltat = TrialFunction(T)
    def solve_forward_and_adjoint(phi):
        # Stokes 1 problem
        solve(e1f(phi)==0, U, bcs=bcs1, solver_parameters=parameters)
        U1.assign(U)
        solve(e2f(phi)==0, U, bcs=bcs2, solver_parameters=parameters)
        U2.assign(U)
        solve(eT==0, t, solver_parameters=parameters)

        # Stokes 2 adjoint
        solve(adjoint(derivative(eT, t, deltat))==-derivative(Jform, t, w), m, solver_parameters=parameters)
        solve(adjoint(derivative(e1f(phi),U)) == -derivative(replace(eT, {w: m}), U1, V) - derivative(Jform, U1, V), XSI, bcs=homogenize(bcs1), solver_parameters=parameters)
        solve(adjoint(derivative(e2f(phi),U)) == (-derivative(replace(eT, {w: m}), U2, V) - derivative(Jform, U2, V)), PSI, bcs=homogenize(bcs2), solver_parameters=parameters)


    solve_forward_and_adjoint(phi)
    Lagr = replace(eT, {w: m}) + replace(e1f(phi), {U: U1, V: XSI}) + replace(e2f(phi), {U: U2, V: PSI}) + Jform
    dL = derivative(Lagr, X)
    dJ = assemble(dL)

    output_dir = "./"
    u1_pvd = File(output_dir + "u1.pvd")
    p1_pvd = File(output_dir + "p1.pvd")
    u2_pvd = File(output_dir + "u2.pvd")
    p2_pvd = File(output_dir + "p2.pvd")
    xsi1_pvd = File(output_dir + "xsi.pvd")
    psi1_pvd = File(output_dir + "psi.pvd")
    t_pvd = File(output_dir + "t.pvd")
    m_pvd = File(output_dir + "m.pvd")
    phi_pvd = File(output_dir + "phi_evo.pvd")
    dJ_pvd = File(output_dir + "dJ_evo.pvd")
    #dJ_pvd.write(dJ)

    u1_func, p1_func = U1.split()
    u2_func, p2_func = U2.split()
    u1_pvd.write(u1_func)
    u2_pvd.write(u2_func)
    p1_pvd.write(p1_func)
    p2_pvd.write(p2_func)
    t_pvd.write(t)


    S = FunctionSpace(mesh, VectorElement("Lagrange", mesh.ufl_cell(), 1))
    theta,xi = [TrialFunction(S), TestFunction( S)]
    beta = Function(S)
    beta_plotting = Function(S)
    dJ_plotting = Function(S)

    av = (Constant(1e3)*inner(grad(theta),grad(xi)) + inner(theta,xi))*(dx) + 1.0e4*(inner(dot(theta,n),dot(xi,n)) * ds)

    bc_beta1 = DirichletBC(S, Constant((0.0, 0.0)), MOUTH1)
    bc_beta2 = DirichletBC(S, Constant((0.0, 0.0)), MOUTH2)

    ## Line search parameters
    alpha0_init,ls,ls_max,gamma,gamma2 = [0.5,0,8,0.1,0.1]
    alpha0 = alpha0_init
    alpha  = alpha0 # Line search step

    ## Stopping criterion parameters
    Nx = 100
    ItMax,It,stop = [int(1.5*Nx), 0, False]

    import numpy as np
    phi_old = Function(PHI)
    Jarr = np.zeros( ItMax )
    beta_pvd = File(output_dir + "beta_evo.pvd")


    hj_solver = HJStabSolver(mesh, PHI, c2_param=1.0)
    reinit_solver = SignedDistanceSolver(mesh, PHI, dt=1e-6)

    hmin = assemble(CellSize(mesh)*dx)
    while It < ItMax and stop == False:

        solve_forward_and_adjoint(phi)

        Lagr = replace(eT, {w: m}) + replace(e1f(phi), {U: U1, V: XSI}) + replace(e2f(phi), {U: U2, V: PSI}) + Jform
        dL = derivative(Lagr, X)

        J = assemble(Jform)
        Jarr[It] = J

        # CFL condition
        maxv = np.max(phi.vector()[:])
        dt = 1e0 * alpha * hmin / maxv
        # ------- LINE SEARCH ------------------------------------------
        if It > 0 and Jarr[It] > Jarr[It-1] and ls < ls_max:
            ls   += 1
            alpha *= gamma
            phi.assign(phi_old)
            phi.assign(hj_solver.solve(beta, phi, steps=3, dt=dt))
            print('Line search iteration : %s' % ls)
            print('Line search step : %.8f' % alpha)
            print('Function value        : %.10f' % Jarr[It])
        else:
            u1_func, p1_func = U1.split()
            u2_func, p2_func = U2.split()
            xsi_func, _ = XSI.split()
            psi_func, _ = PSI.split()
            xsi1_pvd.write(xsi_func)
            psi1_pvd.write(psi_func)
            u1_pvd.write(u1_func)
            u2_pvd.write(u2_func)
            t_pvd.write(t)
            m_pvd.write(m)
            phi_pvd.write(phi)
            print('************ ITERATION NUMBER %s' % It)
            print('Function value        : %.5f' % Jarr[It])
            #print('Compliance            : %.2f' % )
            #print('Volume fraction       : %.2f' % (vol/(lx*ly)))
            # Decrease or increase line search step
            if ls == ls_max: alpha0 = max(alpha0 * gamma2, 0.1*alpha0_init)
            if ls == 0:      alpha0 = min(alpha0 / gamma2, 1)
            # Reset alpha and line search index
            ls,alpha,It = [0,alpha0, It+1]

            assemble(dL, bcs=[bc_beta1, bc_beta2], tensor=dJ)
            dJ_pvd.write(dJ)



            #solve(av==0, beta, bcs=[bc_beta1, bc_beta2], solver_parameters=parameters)
            Av = assemble(av, bcs=[bc_beta1, bc_beta2])
            solve(Av, beta.vector(), dJ)
            #beta_plotting.assign(beta)
            beta_pvd.write(beta)

            phi_old.assign(phi)
            phi.assign(hj_solver.solve(beta, phi, steps=3, dt=dt))

            # Reinit the level set function every five iterations.
            if np.mod(It,5) == 0:
                Dx = hmin
                phi.assign(reinit_solver.solve(phi, Dx))
            #------------ STOPPING CRITERION ---------------------------
            if It>20 and max(abs(Jarr[It-5:It]-Jarr[It-1]))<2.0e-8*Jarr[It-1]/Nx**2/10:
                stop = True




if __name__ == '__main__':
    main()
