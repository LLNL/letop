import firedrake as fd
from firedrake import inner, dot, grad, div, dx, sqrt, ds


def AdvectionDiffusionGLS(
    V: fd.FunctionSpace,
    theta: fd.Function,
    phi: fd.Function,
    phi_t: fd.Function,
    PeInv: fd.Constant = fd.Constant(1e-5),
):
    rho = fd.TestFunction(V)
    F = (
        phi_t * rho * dx
        + (inner(theta, grad(phi)) * rho + PeInv * inner(grad(phi), grad(rho)))
        * dx
    )

    h = fd.CellDiameter(V.ufl_domain())
    R_U = phi_t + dot(theta, grad(phi)) - PeInv * div(grad(phi))
    beta_gls = 0.9
    tau_gls = beta_gls * (
        (4.0 * dot(theta, theta) / h ** 2) + 9.0 * (4.0 * PeInv / h ** 2) ** 2
    ) ** (-0.5)
    degree = 4

    theta_U = dot(theta, grad(rho)) - PeInv * div(grad(rho))
    F += tau_gls * inner(R_U, theta_U) * dx(degree=degree)

    return F
