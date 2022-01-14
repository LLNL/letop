import firedrake as fd
import firedrake_adjoint as fda
from firedrake import inner, grad, dx, dS, jump, tanh, sqrt, sin, cos, pi
from firedrake.petsc import PETSc
import argparse
from pyadjoint.placeholder import Placeholder
from mesh2 import RectangleMesh2, BoxMesh3
from parameters import DIRICHLET_1, DIRICHLET_2
from letop.physics.utils import hs
from letop.levelset import LevelSetFunctional, RegularizationSolver
from letop.optimization import (
    InfDimProblem,
    Constraint,
    nlspace_solve,
    read_checkpoint,
    is_checkpoint,
)


def print(x):
    return PETSc.Sys.Print(x)


def two_electrode():
    parser = argparse.ArgumentParser(description="Level set method parameters")
    parser.add_argument(
        "--tau",
        action="store",
        dest="tau",
        type=float,
        help="tau: k_eL / Sigma_eD ratio",
        default=0.01,
    )
    parser.add_argument(
        "--xi",
        action="store",
        dest="xi",
        type=float,
        help="Applied non-dimensional scanning rate",
        default=0.5,
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        type=str,
        help="Output directory",
        default="./",
    )
    parser.add_argument(
        "--constraint_value",
        action="store",
        dest="constraint_value",
        type=float,
        help="Constraint value: Min energy stored",
        default=0.2,
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        dest="forward",
        help="Perform only forward simulation",
        default=False,
    )
    parser.add_argument(
        "--interdig",
        action="store_true",
        dest="interdig",
        help="Use an interdigitated design as initial estimation",
        default=False,
    )

    args, unknown = parser.parse_known_args()
    print(args)
    output_dir = args.output_dir
    tau = args.tau
    xi = args.xi
    constraint_value = args.constraint_value

    tlimit = 1.0 / xi
    n_steps = 20
    dt_value = tlimit / n_steps
    porosity = 0.2

    mesh = fd.Mesh("./electrode_mesh.msh")
    # mesh_name = "interdigitated_capacitor.msh"
    # ELECTRODE_1 = 1
    # ELECTRODE_2 = 2
    ELECTROLYTE = 3
    # mesh = fd.Mesh(mesh_name)
    # mh = fd.MeshHierarchy(mesh, 1)
    # mesh = mh[-1]
    x, y = fd.SpatialCoordinate(mesh)

    S = fd.VectorFunctionSpace(mesh, "CG", 1)
    s = fd.Function(S, name="deform")
    mesh.coordinates.assign(mesh.coordinates + s)

    PHI = fd.FunctionSpace(mesh, "CG", 1)
    phi = fd.Function(PHI, name="LevelSet")

    ω = 0.3
    phi_expr = sin(y * pi / ω) * cos(x * pi / ω) - fd.Constant(0.2)
    # phi_expr = cos(x * pi / ω) - fd.Constant(0.2)
    with fda.stop_annotating():
        phi.interpolate(-phi_expr)

    if args.interdig:
        with fda.stop_annotating():
            phi.interpolate(fd.Constant(0.1))
            fd.par_loop(
                ("{[i] : 0 <= i < f.dofs}", "f[i, 0] = -0.1"),
                dx(ELECTROLYTE),
                {"f": (phi, fd.WRITE)},
                is_loopy_kernel=True,
            )
            # Smooth a bit
            af, b = fd.TrialFunction(PHI), fd.TestFunction(PHI)
            kappa = fd.Constant(0.0001)
            aF = inner(kappa * grad(af), grad(b)) * dx + af * b * dx
            b = phi * b * dx
            fd.solve(aF == b, phi)

    V = fd.FunctionSpace(mesh, "CG", 1)
    W = V * V
    u, v = fd.TrialFunction(W), fd.TestFunction(W)
    print(f"DOFs: {W.dim()}")

    direct_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    dt = fd.Constant(dt_value)

    # It might be important to let this value be relatively high so the optimizer can "see it"
    lb_electronic = fd.Constant(1e-6)

    epsilon = fd.Constant(100.0)

    def cp(phi):
        return hs(phi, epsilon=epsilon) + 1e-5

    def energy_resistive_step(u_n, u_sol, kappa_hat, sigma_hat):
        return inner(
            kappa_hat * grad(u_n[1] + u_sol[1]),
            grad(u_n[1] + u_sol[1]),
        ) + inner(
            fd.Constant(1.0 / tau) * sigma_hat * grad(u_n[0] + u_sol[0]),
            grad(u_n[0] + u_sol[0]),
        )

    p = fd.Constant(3.0 / 2.0)

    def sigma_hat_func(phi):
        return (fd.Constant(1.0) - porosity) ** (p) * hs(
            phi, epsilon=epsilon
        ) + lb_electronic * hs(-phi, epsilon=epsilon)

    def kappa_hat_func(phi):
        porosity_new = porosity * 0.0736806299
        return porosity_new ** (p) * hs(phi, epsilon=epsilon) + 1.0 * hs(
            -phi, epsilon=epsilon
        )

    def forward(phi, phi_pvd=None, final_stored_cp=None):

        # phi = 0 ----> epsilon = 1
        # phi = 1 ----> epsilon = porosity
        kappa_hat = kappa_hat_func(phi)
        sigma_hat = sigma_hat_func(phi)

        # Potentials function.
        # First component is electronic
        # Second component is ionic
        u_n = fd.Function(W, name="potential")
        a = (
            fd.Constant(tau) * cp(phi) * (u[0] - u[1]) / dt * v[0] * dx
            + inner(sigma_hat * grad(u[0]), grad(v[0])) * dx
            - cp(phi) * (u[0] - u[1]) / dt * v[1] * dx
            + inner(kappa_hat * grad(u[1]), grad(v[1])) * dx
        )
        L = (
            fd.Constant(tau) * cp(phi) * (u_n[0] - u_n[1]) / dt * v[0] * dx
            - cp(phi) * (u_n[0] - u_n[1]) / dt * v[1] * dx
        )

        t = 0.0

        # BC
        ud = fd.Constant(0.0)
        bc1 = fd.DirichletBC(W.sub(1), ud, (DIRICHLET_2,))
        bc2 = fd.DirichletBC(W.sub(0), fd.Constant(0.0), (DIRICHLET_1,))
        bc1.apply(u_n)
        bcs = [bc1, bc2]

        u_sol = fd.Function(W)
        problem = fd.LinearVariationalProblem(a, L, u_sol, bcs=bcs)
        solver = fd.LinearVariationalSolver(
            problem, solver_parameters=direct_parameters
        )

        energy_resistive = 0.0

        for _ in range(n_steps):
            solver.solve()

            energy_resistive += (
                dt_value
                / 2.0
                * fd.assemble(
                    energy_resistive_step(u_n, u_sol, kappa_hat, sigma_hat)
                    * dx
                )
            )

            t += dt_value

            u_n.assign(u_sol)
            vd = t * xi
            ud.assign(fd.Constant(vd))

            if phi_pvd:
                with fda.stop_annotating():
                    phi1, phi2 = u_n.split()
                    phi1.rename("Phi1")
                    phi2.rename("Phi2")
                    phi_pvd.write(phi1, phi2, time=t)

        return u_n, energy_resistive

    initial_potential = fd.File(f"{output_dir}/initial_potential_.pvd")

    initial_stored_energy = fd.File(f"{output_dir}/initial_stored_energy_.pvd")
    u_n, energy_resistive = forward(
        phi, phi_pvd=initial_potential, final_stored_cp=initial_stored_energy
    )
    u_control = fda.Control(u_n)

    energy_cap = fd.assemble(
        fd.Constant(1.0 / 2.0)
        * cp(phi)
        * (u_n[0] - u_n[1])
        * (u_n[0] - u_n[1])
        * dx
    )
    with fda.stop_annotating():
        max_energy = fd.assemble(
            fd.Constant(1.0 / 2.0)
            * fd.Constant(1.0)
            * fd.Constant(1.0)
            * dx(domain=mesh),
            annotate=False,
        )
    print(
        f"Energy resistive: {energy_resistive}, Energy capacitor: {energy_cap},\
             fraction of Max energy: {energy_cap / max_energy}"
    )
    if args.forward:
        exit()

    import itertools

    global_counter1 = itertools.count()
    controls_f = fd.File(f"{output_dir}/geometry_electrode_interdi.pvd")

    energy_resist_func = fd.Function(V, name="Energy Resistance")
    V_vec = fd.VectorFunctionSpace(mesh, "CG", 1)
    electric_field_1 = fd.Function(V_vec, name="Electronic field")
    electric_field_2 = fd.Function(V_vec, name="Ionic field")
    energy_stored = fd.Function(V, name="Energy Stored")
    energy_resist_pvd = fd.File("energy_resistive.pvd")

    def deriv_cb(phi):
        iter = next(global_counter1)
        if iter % 1 == 0:
            with fda.stop_annotating():
                controls_f.write(phi[0])
                kappa_hat = kappa_hat_func(phi[0])
                sigma_hat = sigma_hat_func(phi[0])
                u_n = u_control.tape_value()
                phi1, phi2 = u_n.split()
                phi1.rename("Potential 1")
                phi2.rename("Potential 2")
                electric_field_1.interpolate(
                    fd.Constant(1.0 / tau) * sigma_hat * grad(u_n[0])
                )
                electric_field_2.interpolate(kappa_hat * grad(u_n[1]))
                energy_resist_func.interpolate(
                    energy_resistive_step(u_n, u_n, kappa_hat, sigma_hat)
                )
                energy_stored.interpolate(
                    fd.Constant(1.0 / 2.0)
                    * cp(phi[0])
                    * (u_n[0] - u_n[1])
                    * (u_n[0] - u_n[1])
                )
                energy_resist_pvd.write(
                    energy_resist_func,
                    phi1,
                    phi2,
                    electric_field_1,
                    electric_field_2,
                    energy_stored,
                )

    m = fda.Control(s)

    Jhat = LevelSetFunctional(
        energy_resistive,
        m,
        phi,
        derivative_cb_pre=deriv_cb,
    )

    energy_balance = fda.AdjFloat(1.0) / energy_cap
    Plimit = 1.0 / (constraint_value * max_energy)

    Phat = LevelSetFunctional(energy_balance, m, phi)
    Pcontrol = fda.Control(energy_balance)

    noslip = fd.Constant((0.0, 0.0))
    bc1_vel = fd.DirichletBC(S, noslip, (DIRICHLET_2,))
    bc2_vel = fd.DirichletBC(S, noslip, (DIRICHLET_1,))
    bcs_vel = [bc1_vel, bc2_vel]
    beta_param = 0.1
    reg_solver = RegularizationSolver(
        S,
        mesh,
        beta=beta_param,
        bcs=bcs_vel,
        gamma=1e6,
        dx=dx,
    )

    problem = InfDimProblem(
        Jhat,
        reg_solver,
        ineqconstraints=[
            Constraint(Phat, Plimit, Pcontrol),
        ],
        reinit_distance=0.08,
    )
    dt_hj = 0.05
    params = {
        "alphaC": 1.0,
        "debug": 5,
        "alphaJ": 0.1,
        "dt": dt_hj,
        "K": 0.1,
        "maxit": 100,
        "maxtrials": 10,
        "itnormalisation": 10,
        "tol_merit": 1e-2,  # new merit can be within 1% of the previous merit
        # "normalize_tol" : -1,
    }
    _ = nlspace_solve(
        problem, params, descent_output_dir="./descent_interdigit/"
    )


if __name__ == "__main__":
    two_electrode()
