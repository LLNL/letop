import firedrake as fd
from firedrake import ds, inner


def pressure_drop(w_sol, INLET, OUTLET, pdrop_constr):
    _, p_sol = fd.split(w_sol)

    return fd.assemble(p_sol * ds(INLET) - p_sol * ds(OUTLET)) / pdrop_constr


def heat_flux(w_sol, t, OUTLET):
    u_sol, _ = fd.split(w_sol)
    scale_factor = 5.0
    n = fd.FacetNormal(w_sol.ufl_domain())
    return fd.assemble(
        fd.Constant(-scale_factor) * inner(t * u_sol, n) * ds(OUTLET)
    )
