import firedrake as fd
from letop.physics import (
    AdvectionDiffusionGLS,
    NavierStokesBrinkmannForm,
    NavierStokesBrinkmannSolver,
    InteriorBC,
)


def temperature_solver(
    T, t, beta, PeInv, solver_parameters, *, t1, INLET1, t2, INLET2
):
    F_T = AdvectionDiffusionGLS(T, beta, t, PeInv=PeInv)

    bc1 = fd.DirichletBC(T, t1, INLET1)
    bc2 = fd.DirichletBC(T, t2, INLET2)
    bcs = [bc1, bc2]
    problem_T = fd.NonlinearVariationalProblem(F_T, t, bcs=bcs)
    return fd.NonlinearVariationalSolver(
        problem_T, solver_parameters=solver_parameters
    )


def flow_solver(
    W,
    w_sol,
    no_flow=None,
    phi=None,
    beta_gls=0.9,
    design_domain=None,
    solver_parameters=None,
    brinkmann_penalty=0.0,
    *,
    inflow,
    inlet,
    walls,
    nu,
):
    F = NavierStokesBrinkmannForm(
        W,
        w_sol,
        nu,
        phi=phi,
        brinkmann_penalty=brinkmann_penalty,
        design_domain=design_domain,
        beta_gls=beta_gls,
    )

    noslip = fd.Constant((0.0, 0.0, 0.0))
    bcs_1 = fd.DirichletBC(W.sub(0), noslip, walls)
    bcs_2 = fd.DirichletBC(W.sub(0), inflow, inlet)
    bcs = [bcs_1, bcs_2]
    if no_flow:
        bcs_no_flow = InteriorBC(W.sub(0), noslip, no_flow)
    bcs.append(bcs_no_flow)

    problem = fd.NonlinearVariationalProblem(F, w_sol, bcs=bcs)

    return NavierStokesBrinkmannSolver(
        problem, solver_parameters=solver_parameters
    )


def bc_velocity_profile(type_he, X):
    u_inflow = 1.0
    from parameters_box import (
        inlet_1_coords,
        inlet_2_coords,
        outlet_1_coords,
        R1,
        R2,
    )

    if type_he == "parallel":
        sign_inlet_1 = 1
        sign_inlet_2 = 1
    elif type_he == "counter":
        inlet_1_coords, outlet_1_coords = outlet_1_coords, inlet_1_coords
        sign_inlet_1 = -1
        sign_inlet_2 = 1
    elif type_he == "u_flow":
        inlet_2_coords, outlet_1_coords = outlet_1_coords, inlet_2_coords
        R1, R2 = R2, R1
        sign_inlet_1 = 1
        sign_inlet_2 = -1
    else:
        raise RuntimeError(
            f"type of heat exchanger {type_he} is invalid. \
                                Choose parallel, counter or u_flow"
        )

    x1, y1, z1 = inlet_1_coords
    x, y, z = X
    r1_2 = (x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2
    inflow1 = fd.as_vector(
        [sign_inlet_1 * u_inflow / R1 ** 2 * (R1 ** 2 - r1_2), 0.0, 0.0]
    )
    x2, y2, z2 = inlet_2_coords
    r2_2 = (x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2
    inflow2 = fd.as_vector(
        [sign_inlet_2 * u_inflow / R2 ** 2 * (R2 ** 2 - r2_2), 0.0, 0.0]
    )

    return (
        inflow1,
        inflow2,
    )
