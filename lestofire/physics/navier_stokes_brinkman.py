from cmath import tau
import firedrake as fd
from firedrake import (
    inner,
    grad,
    div,
    dx,
    ds,
    dot,
    dS,
    jump,
    avg,
    Constant,
    exp,
)
from firedrake.function import Function
from pyadjoint.enlisting import Enlist
import ufl
from .utils import hs
from typing import Callable, Union
from ufl.algebra import Product
from functools import partial
from firedrake.cython.dmcommon import FACE_SETS_LABEL, CELL_SETS_LABEL
from pyop2.utils import as_tuple
from firedrake.utils import cached_property
from pyop2.datatypes import IntType
import numpy as np
from typing import List


def mark_no_flow_regions(mesh: fd.Mesh, regions: List, regions_marker: List):
    dm = mesh.topology_dm
    dm.createLabel(FACE_SETS_LABEL)
    dm.markBoundaryFaces("boundary_faces")
    for region, marker in zip(regions, regions_marker):
        cells = dm.getStratumIS(CELL_SETS_LABEL, region)
        for cell in cells.array:
            faces = dm.getCone(cell)
            for face in faces:
                if dm.getLabelValue("boundary_faces", face) == 1:
                    continue
                dm.setLabelValue(FACE_SETS_LABEL, face, marker)
    dm.removeLabel("boundary_faces")
    return mesh


class InteriorBC(fd.DirichletBC):
    @cached_property
    def nodes(self):
        dm = self.function_space().mesh().topology_dm
        section = self.function_space().dm.getDefaultSection()
        nodes = []
        for sd in as_tuple(self.sub_domain):
            nfaces = dm.getStratumSize(FACE_SETS_LABEL, sd)
            faces = dm.getStratumIS(FACE_SETS_LABEL, sd)
            if nfaces == 0:
                continue
            for face in faces.indices:
                # if dm.getLabelValue("interior_facets", face) < 0:
                #    continue
                closure, _ = dm.getTransitiveClosure(face)
                for p in closure:
                    dof = section.getDof(p)
                    offset = section.getOffset(p)
                    nodes.extend((offset + d) for d in range(dof))
        return np.unique(np.asarray(nodes, dtype=IntType))


def NavierStokesBrinkmannForm(
    W: fd.FunctionSpace,
    w: fd.Function,
    nu,
    phi: Union[fd.Function, Product] = None,
    brinkmann_penalty: fd.Constant = None,
    brinkmann_min=0.0,
    design_domain=None,
    hs: Callable = hs,
    beta_gls=0.9,
) -> ufl.form:
    """Returns the Galerkin Least Squares formulation for the Navier-Stokes problem with a Brinkmann term

    Args:
        W (fd.FunctionSpace): [description]
        w (fd.Function): [description]
        phi (fd.Function): [description]
        nu ([type]): [description]
        brinkmann_penalty ([type], optional): [description]. Defaults to None.
        design_domain ([type], optional): Region where the level set is defined. Defaults to None.

    Returns:
        ufl.form: Nonlinear form
    """
    mesh = w.ufl_domain()

    W_elem = W.ufl_element()
    assert isinstance(W_elem, fd.MixedElement)
    if brinkmann_penalty:
        assert isinstance(brinkmann_penalty, fd.Constant)
    assert W_elem.num_sub_elements() == 2

    for W_sub_elem in W_elem.sub_elements():
        assert W_sub_elem.family() == "Lagrange"
        assert W_sub_elem.degree() == 1
    assert isinstance(W_elem.sub_elements()[0], fd.VectorElement)

    v, q = fd.TestFunctions(W)
    u, p = fd.split(w)

    # Main NS form
    F = (
        nu * inner(grad(u), grad(v)) * dx
        + inner(dot(grad(u), u), v) * dx
        - p * div(v) * dx
        + div(u) * q * dx
    )

    # Brinkmann terms for design
    def add_measures(list_dd, **kwargs):
        return sum([dx(dd, kwargs) for dd in list_dd[1::]], dx(list_dd[0]))

    def alpha(phi):
        return brinkmann_penalty * hs(phi) + Constant(brinkmann_min)

    if brinkmann_penalty and phi is not None:
        if design_domain is not None:
            dx_brinkmann = add_measures(Enlist(design_domain))
        else:
            dx_brinkmann = dx

        F = F + alpha(phi) * inner(u, v) * dx_brinkmann

    # GLS stabilization
    R_U = dot(u, grad(u)) - nu * div(grad(u)) + grad(p)
    beta_gls = fd.Constant(beta_gls)
    h = fd.CellSize(mesh)
    tau_gls = beta_gls * (
        (4.0 * dot(u, u) / h ** 2) + 9.0 * (4.0 * nu / h ** 2) ** 2
    ) ** (-0.5)
    degree = 8

    theta_U = dot(u, grad(v)) - nu * div(grad(v)) + grad(q)
    F = F + tau_gls * inner(R_U, theta_U) * dx(degree=degree)

    if brinkmann_penalty and phi is not None:
        tau_gls_alpha = beta_gls * (
            (4.0 * dot(u, u) / h ** 2)
            + 9.0 * (4.0 * nu / h ** 2) ** 2
            + (alpha(phi) / 1.0) ** 2
        ) ** (-0.5)
        R_U_alpha = R_U + alpha(phi) * u
        theta_alpha = theta_U + alpha(phi) * v

        F = F + tau_gls_alpha * inner(R_U_alpha, theta_alpha) * dx_brinkmann(
            degree=degree
        )
        if (
            design_domain is not None
        ):  # Substract this domain from the original integral
            F = F - tau_gls * inner(R_U, theta_U) * dx_brinkmann(degree=degree)

    return F


class NavierStokesBrinkmannSolver(object):
    def __init__(
        self, problem: fd.NonlinearVariationalProblem, **kwargs
    ) -> None:
        """Same than NonlinearVariationalSolver, but with just the SIMPLE preconditioner by default
        Args:
            problem ([type]): [description]
            nullspace ([type], optional): [description]. Defaults to None.
            solver_parameters ([type], optional): [description]. Defaults to None.
        """
        solver_parameters_default = {
            # "snes_type": "ksponly",
            # "snes_no_convergence_test" : None,
            # "snes_max_it": 1,
            "snes_type": "newtonls",
            "snes_linesearch_type": "l2",
            "snes_linesearch_maxstep": 1.0,
            # "snes_monitor": None,
            # "snes_linesearch_monitor": None,
            "snes_rtol": 1.0e-4,
            "snes_atol": 1.0e-4,
            "snes_stol": 0.0,
            "snes_max_linear_solve_fail": 10,
            "snes_converged_reason": None,
            "ksp_type": "fgmres",
            "mat_type": "aij",
            # "default_sub_matrix_type": "aij",
            "ksp_rtol": 1.0e-4,
            "ksp_atol": 1.0e-4,
            "ksp_max_it": 2000,
            # "ksp_monitor": None,
            "ksp_converged_reason": None,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_factorization_type": "full",
            "pc_fieldsplit_schur_precondition": "selfp",
            "fieldsplit_0": {
                "ksp_type": "richardson",
                "ksp_richardson_self_scale": False,
                "ksp_max_it": 1,
                "pc_type": "ml",
                "ksp_atol": 1e-2,
                "pc_mg_cycle_type": "v",
                "pc_mg_type": "full",
                # "ksp_converged_reason": None,
                # "ksp_monitor": None,
            },
            "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_type": "ml",
                # "ksp_monitor": None,
                # "ksp_converged_reason": None,
            },
            "fieldsplit_1_upper_ksp_type": "preonly",
            "fieldsplit_1_upper_pc_type": "jacobi",
        }
        solver_parameters = kwargs.pop("solver_parameters", None)
        if solver_parameters:
            solver_parameters_default.update(solver_parameters)

        self.solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=solver_parameters_default, **kwargs
        )

    def solve(self):
        self.solver.solve()
