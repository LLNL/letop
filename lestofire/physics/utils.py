import firedrake as fd
import ufl


def hs(phi: fd.Function, epsilon=fd.Constant(10000.0)):
    """Heaviside approximation

    Args:
        phi (fd.Function): Level set
        epsilon ([type], optional): Parameter to approximate the Heaviside. Defaults to Constant(10000.0).

    Returns:
        [type]: [description]
    """
    return fd.Constant(1.0) / (fd.Constant(1.0) + ufl.exp(-epsilon * phi))