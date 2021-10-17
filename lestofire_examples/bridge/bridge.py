import firedrake as fd
from firedrake import (
    cos,
    pi,
    inner,
    grad,
    dx,
    ds,
    sym,
    nabla_grad,
    tr,
    Identity,
    ds_t,
)
import firedrake_adjoint as fda
import numpy as np

from lestofire.levelset import LevelSetFunctional, RegularizationSolver
from lestofire.optimization import InfDimProblem, Constraint
from lestofire.physics import hs
from lestofire.optimization import nlspace_solve
import itertools
from firedrake import PETSc
from solver_parameters import (
    gamg_parameters,
    hj_solver_parameters,
    reinit_solver_parameters,
)


import argparse

fd.parameters["pyop2_options"]["block_sparsity"] = False


def print(x):
    PETSc.Sys.Print(x)


class MyBC(fd.DirichletBC):
    def __init__(self, V, value, markers):
        # Call superclass init
        # We provide a dummy subdomain id.
        super(MyBC, self).__init__(V, value, 0)
        # Override the "nodes" property which says where the boundary
        # condition is to be applied.
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos > 0)[0])


def create_function_marker(PHI, W, xlimits, ylimits):
    x, y, z = fd.SpatialCoordinate(PHI.ufl_domain())
    x_func, y_func, z_func = (
        fd.Function(PHI),
        fd.Function(PHI),
        fd.Function(PHI),
    )
    with fda.stop_annotating():
        x_func.interpolate(x)
        y_func.interpolate(y)
        z_func.interpolate(z)

    domain = "{[i, j]: 0 <= i < f.dofs and 0<= j <= 3}"
    instruction = f"""
    f[i, j] = 1.0 if (x[i, 0] < {xlimits[1]} and x[i, 0] > {xlimits[0]}) and (y[i, 0] < {ylimits[0]} or y[i, 0] > {ylimits[1]}) and z[i, 0] < 1e-7 else 0.0
    """
    I_BC = fd.Function(W)
    fd.par_loop(
        (domain, instruction),
        dx,
        {
            "f": (I_BC, fd.RW),
            "x": (x_func, fd.READ),
            "y": (y_func, fd.READ),
            "z": (z_func, fd.READ),
        },
        is_loopy_kernel=True,
    )

    return I_BC


def compliance_bridge():

    parser = argparse.ArgumentParser(description="Heat exchanger")
    parser.add_argument(
        "--n_iters",
        dest="n_iters",
        type=int,
        action="store",
        default=1000,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        action="store",
        default="./",
        help="Output directory",
    )
    opts = parser.parse_args()
    output_dir = opts.output_dir
    # Elasticity parameters
    E, nu = 1.0, 0.3
    rho_min = fd.Constant(1e-4)  # Min Vol fraction
    eps = fd.Constant(100.0)  # Heaviside parameter
    mu, lmbda = fd.Constant(E / (2 * (1 + nu))), fd.Constant(
        E * nu / ((1 + nu) * (1 - 2 * nu))
    )

    mesh = fd.RectangleMesh(10, 20, 0.5, 1, quadrilateral=True)
    mh = fd.MeshHierarchy(mesh, 1)
    m = fd.ExtrudedMeshHierarchy(mh, height=1, base_layer=20)
    mesh = m[-1]

    S = fd.VectorFunctionSpace(mesh, "CG", 1)
    s = fd.Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    x, y, z = fd.SpatialCoordinate(mesh)
    PHI = fd.FunctionSpace(mesh, "CG", 1)
    lx = 1.0
    ly = 2.0
    lz = ly
    phi_expr = (
        -cos(4.0 / lx * pi * x)
        * cos(4.0 * pi / ly * y)
        * cos(4.0 / lz * pi * z)
        - 0.6
    )
    with fda.stop_annotating():
        phi = fd.interpolate(-phi_expr, PHI)
        phi.rename("LevelSet")

    H1 = fd.VectorElement("CG", mesh.ufl_cell(), 1)
    W = fd.FunctionSpace(mesh, H1)
    print(f"DOFS: {W.dim()}")

    modes = [fd.Function(W) for _ in range(6)]
    modes[0].interpolate(fd.Constant([1, 0, 0]))
    modes[1].interpolate(fd.Constant([0, 1, 0]))
    modes[2].interpolate(fd.Constant([0, 0, 1]))
    modes[3].interpolate(fd.as_vector([0, z, -y]))
    modes[4].interpolate(fd.as_vector([-z, 0, x]))
    modes[5].interpolate(fd.as_vector([y, -x, 0]))
    nullmodes = fd.VectorSpaceBasis(modes)
    # Make sure they're orthonormal.
    nullmodes.orthonormalize()

    u = fd.TrialFunction(W)
    v = fd.TestFunction(W)

    def epsilon(u):
        return sym(nabla_grad(u))

    def sigma(v):
        return 2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(3)

    # Variational forms
    a = inner(hs(phi, eps, min_value=rho_min) * sigma(u), nabla_grad(v)) * dx(
        degree=2
    )
    t = fd.Constant((0.0, 0.0, -1.0e-1))
    L = inner(t, v) * ds_t

    # Dirichlet BCs
    ylimits = (0.2, 1.8)
    xlimits = (0.4, 0.6)
    I_BC = create_function_marker(PHI, W, xlimits, ylimits)
    bc1 = MyBC(W, 0, I_BC)
    bc2 = fd.DirichletBC(W.sub(0), fd.Constant(0.0), 2)
    bc3 = fd.DirichletBC(W.sub(1), fd.Constant(0.0), 4)

    u_sol = fd.Function(W)
    fd.solve(
        a == L,
        u_sol,
        bcs=[bc1, bc2, bc3],
        solver_parameters=gamg_parameters,
        near_nullspace=nullmodes,
    )

    # Cost function
    Jform = fd.assemble(
        inner(hs(phi, eps, min_value=rho_min) * sigma(u_sol), epsilon(u_sol))
        * dx(degree=2)
    )
    # Constraint
    VolPen = fd.assemble(hs(phi, eps, min_value=rho_min) * dx(degree=2))
    total_vol = fd.assemble(fd.Constant(1.0) * dx(domain=mesh), annotate=False)
    VolControl = fda.Control(VolPen)
    Vval = 0.15 * total_vol

    # Plotting
    global_counter1 = itertools.count()
    phi_pvd = fd.File(f"{output_dir}/level_set_evolution.pvd")

    def deriv_cb(phi):
        iter = next(global_counter1)
        if iter % 10 == 0:
            phi_pvd.write(phi[0])

    c = fda.Control(s)
    Jhat = LevelSetFunctional(Jform, c, phi, derivative_cb_pre=deriv_cb)
    Vhat = LevelSetFunctional(VolPen, c, phi)

    # Regularization solver. Zero on the BCs boundaries
    beta_param = 0.01
    bcs_vel_1 = MyBC(S, 0, I_BC)
    bcs_vel_2 = fd.DirichletBC(S, fd.Constant((0.0, 0.0, 0.0)), "top")
    bcs_vel = [bcs_vel_1, bcs_vel_2]
    reg_solver = RegularizationSolver(
        S,
        mesh,
        beta=beta_param,
        gamma=1.0e4,
        dx=dx,
        bcs=bcs_vel,
        output_dir=None,
        solver_parameters=gamg_parameters,
    )
    dt = 0.1
    tol = 1e-5

    params = {
        "alphaC": 1.0,
        "K": 0.01,
        "debug": 5,
        "maxit": opts.n_iters,
        "alphaJ": 2.0,
        "dt": dt,
        "maxtrials": 500,
        "tol_merit": 5e-2,  # new merit can be within 0.5% of the previous merit
        "itnormalisation": 50,
        "tol": tol,
    }
    hj_solver_parameters["ts_dt"] = dt / 50.0
    solver_parameters = {
        "hj_solver": hj_solver_parameters,
        "reinit_solver": reinit_solver_parameters,
    }

    vol_constraint = Constraint(Vhat, Vval, VolControl)
    problem = InfDimProblem(
        Jhat,
        reg_solver,
        ineqconstraints=vol_constraint,
        solver_parameters=solver_parameters,
    )
    _ = nlspace_solve(problem, params)


if __name__ == "__main__":
    compliance_bridge()
