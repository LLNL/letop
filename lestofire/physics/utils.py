import firedrake as fd
from functools import lru_cache
import numpy as np
import ufl


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


def hs(phi: fd.Function, epsilon=fd.Constant(10000.0)):
    """Heaviside approximation

    Args:
        phi (fd.Function): Level set
        epsilon ([type], optional): Parameter to approximate the Heaviside. Defaults to Constant(10000.0).

    Returns:
        [type]: [description]
    """
    return fd.Constant(1.0) / (fd.Constant(1.0) + ufl.exp(-epsilon * phi))