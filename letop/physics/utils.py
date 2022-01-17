import firedrake as fd
from functools import lru_cache
import numpy as np
import ufl
import math
from mpi4py import MPI


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
        fd.CellDiameter(mesh)
        / fd.CellVolume(mesh)
        * fd.TestFunction(DG0)
        * fd.dx
    ).dat.data_ro
    local_min_size = np.max(h_sizes)
    return mesh.comm.allreduce(local_min_size, op=MPI.MAX)


def hs(
    phi: fd.Function,
    epsilon: fd.Constant = fd.Constant(10000.0),
    width_h: float = None,
    shift: float = 0.0,
    min_value: fd.Function = fd.Constant(0.0),
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
        [type]: Heavise function onto phi, i.e. 1 for phi > 0 and 0 for phi < 0
    """
    if width_h:
        if epsilon:
            fd.warning(
                "Epsilon and width_h are both defined, pick one or the other. \
                Overriding epsilon choice"
            )
        mesh = phi.ufl_domain()
        hmin = min_mesh_size(mesh)
        epsilon = fd.Constant(
            math.log(0.99 ** 2 / 0.01 ** 2) / (width_h * hmin)
        )

    return (
        fd.Constant(1.0)
        / (fd.Constant(1.0) + ufl.exp(-epsilon * (phi - fd.Constant(shift))))
        + min_value
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
        epsilon = fd.Constant(
            math.log(0.95 ** 2 / 0.05 ** 2) / (width_h * hmin)
        )

    return (
        fd.Constant(epsilon)
        * ufl.exp(-epsilon * phi)
        / (fd.Constant(1.0) + ufl.exp(-epsilon * phi)) ** 2
    )


def calculate_max_vel(velocity: fd.Function):
    mesh = velocity.ufl_domain()
    MAXSP = fd.FunctionSpace(mesh, "R", 0)
    maxv = fd.Function(MAXSP)
    domain = "{[i, j] : 0 <= i < u.dofs}"
    instruction = """
                    maxv[0] = abs(u[i, 0]) + abs(u[i, 1])
                    """
    fd.par_loop(
        (domain, instruction),
        fd.dx,
        {"u": (velocity, fd.READ), "maxv": (maxv, fd.MAX)},
        is_loopy_kernel=True,
    )
    maxval = maxv.dat.data[0]
    return maxval


@lru_cache(1)
def max_mesh_dimension(mesh: fd.Mesh):
    coords = mesh.coordinates
    MAXSP = fd.FunctionSpace(mesh, "R", 0)
    max_y = fd.Function(MAXSP)
    min_y = fd.Function(MAXSP)
    max_x = fd.Function(MAXSP)
    min_x = fd.Function(MAXSP)
    domain = "{[i, j] : 0 <= i < u.dofs}"

    def extract_comp(mode, comp, result):
        instruction = f"""
                       component[0] = abs(u[i, {comp}])
                       """
        fd.par_loop(
            (domain, instruction),
            fd.dx,
            {"u": (coords, fd.READ), "component": (result, mode)},
            is_loopy_kernel=True,
        )
        return result

    max_y_comp = extract_comp(fd.MAX, 1, max_y).dat.data[0]
    min_y_comp = extract_comp(fd.MIN, 1, min_y).dat.data[0]
    max_x_comp = extract_comp(fd.MAX, 0, max_x).dat.data[0]
    min_x_comp = extract_comp(fd.MIN, 0, min_x).dat.data[0]
    max_dim_x = abs(max_x_comp - min_x_comp)
    max_dim_y = abs(max_y_comp - min_y_comp)

    max_dim = max(max_dim_x, max_dim_y)
    if mesh.geometric_dimension() == 3:
        max_z = fd.Function(MAXSP)
        min_z = fd.Function(MAXSP)
        max_z_comp = extract_comp(fd.MAX, 2, max_z).dat.data[0]
        min_z_comp = extract_comp(fd.MIN, 2, min_z).dat.data[0]
        max_dim_z = abs(max_z_comp - min_z_comp)
        max_dim = max(max_dim, max_dim_z)

    return max_dim
