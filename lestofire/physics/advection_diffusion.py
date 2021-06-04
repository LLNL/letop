import firedrake as fd
from firedrake import inner, dot, grad, div, dx, sqrt


def AdvectionDiffusionSUPG(V: fd.FunctionSpace, theta, k, phi, phi_t):
    rho = fd.TestFunction(V)
    cell_type = V.ufl_domain().ufl_coordinate_element().cell().cellname()
    if cell_type in ["triangle", "tetrahedron"]:
        h = fd.CellSize(V.ufl_domain())
    elif cell_type == "quadrilateral":
        h = fd.CellDiameter(V.ufl_domain())
    else:
        raise RuntimeError(f"Element {cell_type} not supported")
    F = (
        phi_t * rho * dx
        + (inner(theta, grad(phi)) * rho + k * inner(grad(phi), grad(rho)))
        * dx
    )

    # Strong form esidual
    r = phi_t + dot(theta, grad(phi)) - k * div(grad(phi))
    # Add SUPG stabilisation terms
    vnorm = sqrt(dot(theta, theta)) + 1e-8
    F += (h / (2.0 * vnorm)) * dot(theta, grad(rho)) * r * dx
    return F
