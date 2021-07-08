import firedrake as fd
from firedrake import inner, dot, grad, div, dx, sqrt, ds


def AdvectionDiffusionGLS(
    V: fd.FunctionSpace,
    theta: fd.Function,
    phi: fd.Function,
    phi_t: fd.Function,
    PeInv: float = 1e-4,
):
    PeInv_ct = fd.Constant(PeInv)
    rho = fd.TestFunction(V)
    F = (
        phi_t * rho * dx
        + (
            inner(theta, grad(phi)) * rho
            + PeInv_ct * inner(grad(phi), grad(rho))
        )
        * dx
    )

    h = fd.CellDiameter(V.ufl_domain())
    R_U = phi_t + dot(theta, grad(phi)) - PeInv_ct * div(grad(phi))
    beta_gls = 0.9
    tau_gls = beta_gls * (
        (4.0 * dot(theta, theta) / h ** 2)
        + 9.0 * (4.0 * PeInv_ct / h ** 2) ** 2
    ) ** (-0.5)
    degree = 4

    theta_U = dot(theta, grad(rho)) - PeInv_ct * div(grad(rho))
    F += tau_gls * inner(R_U, theta_U) * dx(degree=degree)

    return F
