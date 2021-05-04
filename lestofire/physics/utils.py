import firedrake as fd
from functools import lru_cache
import numpy as np
import ufl
import math


@lru_cache(1)
def min_mesh_size(mesh):
    """Calculate minimum cell diameter in mesh

    Args:
        mesh ([type]): [description]

    Returns:
        float: [description]
    """
    DG0 = fd.FunctionSpace(mesh, "DG", 0)
    h_sizes = fd.assemble(
        fd.CellDiameter(mesh) / fd.CellVolume(mesh) * fd.TestFunction(DG0) * fd.dx
    ).dat.data_ro
    local_min_size = np.max(h_sizes)
    return local_min_size


def hs(
    phi: fd.Function,
    epsilon: fd.Constant = fd.Constant(10000.0),
    width_h: float = None,
    shift: float = 0.0,
):
    """Heaviside approximation

    Args:
        phi (fd.Function): Level set
        epsilon ([type], optional): Parameter to approximate the Heaviside.
        Defaults to Constant(10000.0).
        width_h (float): Width of the Heaviside approximation transition in
        terms of multiple of the mesh element size
        shift: (float): Shift the level set value to define the interface.

    Returns:
        [type]: [description]
    """
    if width_h:
        if epsilon:
            fd.warning(
                "Epsilon and width_h are both defined, pick one or the other. \
                Overriding epsilon choice"
            )
        mesh = phi.ufl_domain()
        hmin = min_mesh_size(mesh)
        epsilon = fd.Constant(math.log(0.99 ** 2 / 0.01 ** 2) / (width_h * hmin))

    return fd.Constant(1.0) / (
        fd.Constant(1.0) + ufl.exp(-epsilon * (phi - fd.Constant(shift)))
    )


def dirac_delta(phi: fd.Function, epsilon=fd.Constant(10000.0), width_h=None):
    """Dirac delta approximation

    Args:
        phi (fd.Function): Level set
        epsilon ([type], optional): Parameter to approximate the Heaviside.
        Defaults to Constant(10000.0).
        width_h (float): Width of the Heaviside approximation transition in
        terms of multiple of the mesh element size

    Returns:
        [type]: [description]
    """
    if width_h:
        if epsilon:
            fd.warning(
                "Epsilon and width_h are both defined, pick one or the other. \
                Overriding epsilon choice"
            )
        mesh = phi.ufl_domain()
        hmin = min_mesh_size(mesh)
        epsilon = fd.Constant(math.log(0.95 ** 2 / 0.05 ** 2) / (width_h * hmin))

    return (
        fd.Constant(epsilon)
        * ufl.exp(-epsilon * phi)
        / (fd.Constant(1.0) + ufl.exp(-epsilon * phi)) ** 2
    )
