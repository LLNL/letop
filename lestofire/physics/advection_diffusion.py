import firedrake as fd
from firedrake import inner, dot, grad, div, dx, sqrt


def AdvectionSUPG(
    V: fd.FunctionSpace,
    theta: fd.Function,
    phi: fd.Function,
    phi_t: fd.Function,
):
    rho = fd.TestFunction(V)
    cell_type = V.ufl_domain().ufl_coordinate_element().cell().cellname()
    if cell_type in ["triangle", "tetrahedron"]:
        h = fd.CellSize(V.ufl_domain())
    elif cell_type == "quadrilateral":
        h = fd.CellDiameter(V.ufl_domain())
    else:
        raise RuntimeError(f"Element {cell_type} not supported")
    F = phi_t * rho * dx + (inner(theta, grad(phi)) * rho) * dx

    # Strong form esidual
    r = phi_t + dot(theta, grad(phi))
    # Add SUPG stabilisation terms
    vnorm = sqrt(dot(theta, theta)) + 1e-3
    F += (h / (2.0 * vnorm)) * dot(theta, grad(rho)) * r * dx
    return F
