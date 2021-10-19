import numpy as np
import cvxopt as cvx
import time
from mpi4py import MPI

import firedrake as fd
from firedrake import inner, Function, dx, Constant, File
from numpy.core.numeric import indices
from sympy import solve
from lestofire.optimization import InfDimProblem
from pyadjoint import no_annotations
from firedrake import PETSc
from functools import partial


def print(x):
    return PETSc.Sys.Print(x)


class MPITimer:
    def __init__(self, comm) -> None:
        self.comm = comm

    def __enter__(self):
        self.start = time.process_time()
        return self

    def __exit__(self, *args):
        self.end = time.process_time()
        total_time = self.end - self.start
        self.min_time = self.comm.allreduce(total_time, op=MPI.MIN)
        self.max_time = self.comm.allreduce(total_time, op=MPI.MAX)


def set_parameters(params):

    params_default = {
        "alphaJ": 1,
        "alphaC": 1,
        "maxit": 4000,
        "maxtrials": 3,
        "debug": 0,
        "normalisation_norm": np.inf,
        "tol_merit": 0,
        "K": 0.1,
        "tol_qp": 1e-20,
        "show_progress_qp": False,
        "monitor_time": False,
        "dt": 1.0,
        "tol": 1e-5,
        "itnormalisation": 10,
    }
    if params is not None:
        for key in params.keys():
            if key not in params_default.keys():
                raise ValueError(
                    "Don't know how to deal with parameter %s (a %s)"
                    % (key, params[key].__class__)
                )

        for (prop, default) in params_default.items():
            params[prop] = params.get(prop, default)
    else:
        params = params_default

    return params


def display(message, debug, level=0, color=None):
    if color:
        try:
            import colored as col

            message = col.stylize(message, col.fg(color))
        except Exception:
            pass
    if debug >= level:
        print(message)


def inner_product(x, y):
    return fd.assemble(inner(x, y) * dx, annotate=False)


def compute_norm(f, norm_type=np.inf):
    if norm_type == np.inf:
        with f.dat.vec_ro as vu:
            return vu.norm(PETSc.NormType.NORM_INFINITY)
    elif norm_type == 2:
        return fd.norm(f)


def getTilde(C, p, eps=0):
    """
    Obtain the set of violated inequality constraints
    """
    tildeEps = C[p:] >= -eps
    tildeEps = np.asarray(np.concatenate(([True] * p, tildeEps)), dtype=bool)
    return tildeEps


def getEps(dC, p, dt, K, norm_type=np.inf):
    if len(dC) == 0:
        return np.array([])
    if norm_type == np.inf:
        eps = []
        for dCi in dC[p:]:
            with dCi.dat.vec_ro as dCv:
                norm_inf = dCv.norm(PETSc.NormType.NORM_INFINITY)
                eps.append(norm_inf)
        eps = np.array(eps) * dt * K
    elif norm_type == 2:
        eps = np.array([fd.norm(dCi) * dt * K for dCi in dC[p:]])
    return eps


def p_matrix_eval(dC, tildeEps):
    p_matrix = np.zeros((sum(tildeEps), sum(tildeEps)))
    ii = 0
    for i, tildei in enumerate(tildeEps):
        jj = 0
        for j, tildej in enumerate(tildeEps):
            if tildej and tildei:
                p_matrix[ii, jj] = inner_product(dC[i], dC[j])
                jj += 1
        if tildei:
            ii += 1
    return p_matrix


def q_vector_eval(dJ, dC, tildeEps):
    q_vector = np.zeros((sum(tildeEps), 1))
    ii = 0
    for i, tildei in enumerate(tildeEps):
        if tildei:
            q_vector[ii] = inner_product(dJ, dC[i])
            ii += 1
    return q_vector


def invert_dCdCT(dCdCT, debug):
    try:
        dCtdCtTinv = np.linalg.inv(dCdCT)
    except Exception:
        display(
            "Warning, constraints are not qualified. "
            + "Using pseudo-inverse.",
            debug,
            1,
            color="orange_4a",
        )
        dCtdCtTinv = np.linalg.pinv(dCdCT)

    return dCtdCtTinv


def line_search(
    problem,
    orig_phi,
    new_phi,
    merit_eval_new,
    merit,
    AJ,
    AC,
    dt=1.0,
    maxtrials=10,
    tol_merit=1e-3,
    debug=1.0,
):

    vel_scale = problem.velocity_scale(problem.delta_x)
    problem.delta_x /= vel_scale
    success = 0
    for k in range(maxtrials):
        new_phi.assign(
            problem.retract(orig_phi, problem.delta_x, scaling=(dt * 0.5 ** k))
        )
        (newJ, newG, newH) = problem.eval(new_phi)
        newC = np.concatenate((newG, newH))
        new_merit = merit_eval_new(
            AJ,
            newJ,
            AC,
            newC,
        )
        print(f"newJ={newJ}, newC={newC}")
        print(f"merit={merit}, new_merit={new_merit}")
        if new_merit < (1 + np.sign(merit) * tol_merit) * merit:
            success = 1
            break
        else:
            display(
                "Warning, merit function did not decrease "
                + f"(merit={merit}, new_merit={new_merit})"
                + f"-> Trial {k+1}",
                debug,
                0,
                "red",
            )
            # Last attempt is accepted, forgo reset the distance.
            if k == maxtrials - 1:
                break
            problem.reset_distance()
    if not success:
        display(
            "All trials have failed, passing to the next iteration.",
            debug,
            color="red",
        )

    return new_phi, newJ, newG, newH


def solve_dual_problem(
    p_matrix, q_vector, tildeEps, p, show_progress_qp=None, tol_qp=None
):
    # Solve the dual problem to obtain the new set of active constraints
    # Solve it using a quadratic solver:
    # minimize    (1/2)*x'*P*x + q'*x
    # subject to  G*x <= h
    #            A*x = b.
    # A and b are optionals and not given here.
    # sum(tildeEps) = p + qtildeEps

    qtildeEps = sum(tildeEps) - p

    Pcvx = cvx.matrix(p_matrix)
    qcvx = cvx.matrix(q_vector)
    Gcvx = cvx.matrix(
        np.concatenate((np.zeros((qtildeEps, p)), -np.eye(qtildeEps)), axis=1)
    )
    hcvx = cvx.matrix(np.zeros((qtildeEps, 1)))
    if p + qtildeEps > 0:
        return cvx.solvers.qp(
            Pcvx,
            qcvx,
            Gcvx,
            hcvx,
            options={
                "show_progress": show_progress_qp,
                "reltol": 1e-20,
                "abstol": 1e-20,
                "feastol": tol_qp,
            },
        )
    else:
        return None


def dCdCT_eval(dC, hat):
    dCdCT = np.zeros((sum(hat), sum(hat)))
    ii = 0
    for i, tildei in enumerate(hat):
        jj = 0
        for j, tildej in enumerate(hat):
            if tildej and tildei:
                dCdCT[ii, jj] = inner_product(dC[i], dC[j])
                jj += 1
        if tildei:
            ii += 1
    return dCdCT


def dCdCT_eval_tilde(dC, indicesEps):
    dCdCT = np.zeros((sum(indicesEps), sum(indicesEps)))
    ii = 0
    for i, indi in enumerate(indicesEps):
        jj = 0
        for j, indj in enumerate(indicesEps):
            if indi and indj:
                dCdCT[ii, jj] = inner_product(dC[i], dC[j])
                jj += 1
        if indi:
            ii += 1
    return dCdCT


def dCdJ_eval(dJ, dC, hat):
    dCdJ = hat.copy()
    ii = 0
    for i, hati in enumerate(hat):
        if hati:
            dCdJ[ii] = inner_product(dC[i], dJ)
            ii += 1
    return dCdJ


def xiJ_eval(dJ, dC, muls, hat):
    xiJ = Function(dJ.function_space())
    if hat.any():
        list_func = []
        for i, hati in enumerate(hat):
            if hati:
                list_func.append(Constant(muls[i]) * dC[i])
            else:
                # Still add a function to be able to call Function.assign with
                # objects of same shape.
                list_func.append(Constant(0.0) * dC[i])
        xiJ.assign(dJ + sum(list_func))
    else:
        xiJ.assign(dJ)

    return xiJ


def xiC_eval(C, dC, dCtdCtTinv, alphas, indicesEps):
    if len(C) == 0:
        return None
    # Sum of functions over indicesEps
    xiC = Function(dC[0].function_space())
    if indicesEps.any():
        list_func = []
        dCdTinvCalpha = dCtdCtTinv.dot(C[indicesEps] * alphas[indicesEps])
        ii = 0
        for i, indi in enumerate(indicesEps):
            if indi:
                list_func.append(Constant(dCdTinvCalpha[ii]) * dC[i])
                ii += 1
        xiC.assign(sum(list_func))
    return xiC


def merit_eval(muls, indicesEps, dCtdCtTinv, AJ, J, AC, C):
    return AJ * (J + muls.dot(C)) + 0.5 * AC * C[indicesEps].dot(
        dCtdCtTinv.dot(C[indicesEps])
    )


@no_annotations
def nlspace_solve(
    problem: InfDimProblem, params=None, results=None, descent_output_dir=None
):
    """
    Solve the optimization problem
        min      J(phi)
        phi in V
        under the constraints
        g_i(phi)=0  for all i=0..p-1
        h_i(phi)<=0 for all i=0..q-1

    Usage
    -----
    results=nlspace_solve(problem: InfDimProblem, params: dict, results:dict)

    Inputs
    ------
    problem : an `~InfDimProblem` object corresponding to the optimization
                  problem above.

    params  : (optional) a dictionary containing algorithm parameters
              (see below).

    results : (optional) a previous output of the nlspace_solve` function.
              The optimization will keep going from the last input of
              the dictionary `results['phi'][-1]`.
              Useful to restart an optimization after an interruption.
    descent_output_dir : Plot the descent direction in the given directory

    Output
    ------
    results : dictionary containing
        results['J']       : values of the objective function along the path
                             (J(phi_0),...,J(phi_n))
        results['G']       : equality constraint values
                             (G(phi_0),...,G(phi_n))
        results['H']       : inequality constraints values
                             (H(phi_0),...,H(phi_n))
        results['muls']    : lagrange multiplier values
                             (mu(phi_0),...,mu(phi_n))


    Optional algorithm parameters
    -----------------------------

    params['alphaJ']   : (default 1) scaling coefficient for the null space
        step xiJ decreasing the objective function

    params['alphaC']   : (default 1) scaling coefficient for the Gauss Newton
        step xiC decreasing the violation of the constraints

    params['alphas']   : (optional) vector of dimension
        problem.nconstraints + problem.nineqconstraints containing
        proportionality coefficients scaling the Gauss Newton direction xiC for
        each of the constraints

    params['debug'] : Tune the verbosity of the output (default 0)
                      Set param['debug']=-1 to display only the final result
                      Set param['debug']=-2 to remove any output

    params['dt'] : (default : `1.0`). Pseudo time-step expressed in a time unit.
        Used to modulate the optimization convergence/oscillatory behavior.

    params['hmin'] : (default : `1.0`). Mesh minimum length. TODO Replace this for dt
        in the calculation of the tolerances `eps`

    params['K']: tunes the distance at which inactive inequality constraints
        are felt. Constraints are felt from a distance K*params['dt']

    params['maxit']    : Maximal number of iterations (default : 4000)

    params['maxtrials']: (default 3) number of trials in between time steps
        until the merit function decreases

    params['tol']      : (default 1e-7) Algorithm stops when
            ||phi_{n+1}-phi_n||<params['tol']
        or after params['maxit'] iterations.

    params['tol_merit'] : (default 0) a new iterate phi_{n+1} is accepted if
        merit(phi_{n+1})<(1+sign(merit(phi_n)*params['tol_merit']))*merit(phi_n)

    params['tol_qp'] : (default 1e-20) the tolerance for the qp solver cvxopt

    params['show_progress_qp'] : (default False) If true, then the output of
        cvxopt will be displayed between iterations.

    params['monitor_time'] : (default False) If true, then the output of
        the time taken between optimization iterations.

    """

    params = set_parameters(params)

    alphas = np.asarray(
        params.get(
            "alphas",
            [1] * (len(problem.eqconstraints) + len(problem.ineqconstraints)),
        )
    )

    if descent_output_dir:
        descent_pvd = File(f"{descent_output_dir}/descent_direction.pvd")

    results = {
        "phi": [],
        "J": [],
        "G": [],
        "H": [],
        "muls": [],
        "merit": [],
    }

    phi = problem.x0()

    (J, G, H) = problem.eval(phi)

    normdx = 1  # current value for x_{n+1}-x_n

    new_phi = Function(phi.function_space())
    orig_phi = Function(phi.function_space())
    while normdx > params["tol"] and len(results["J"]) < params["maxit"]:
        with MPITimer(phi.comm) as timings:

            results["J"].append(J)
            results["G"].append(G)
            results["H"].append(H)

            if problem.accept():
                break

            it = len(results["J"]) - 1
            display("\n", params["debug"], 1)
            display(
                f"{it}. J="
                + format(J, ".4g")
                + " "
                + "G=["
                + ",".join(format(phi, ".4g") for phi in G[:10])
                + "] "
                + "H=["
                + ",".join(format(phi, ".4g") for phi in H[:10])
                + "] "
                + " ||dx||_V="
                + format(normdx, ".4g"),
                params["debug"],
                0,
            )

            # Returns the gradients (in the primal space). They are
            # firedrake.Function's
            (dJ, dG, dH) = problem.eval_gradients(phi)
            dC = dG + dH

            H = np.asarray(H)
            G = np.asarray(G)
            C = np.concatenate((G, H))

            # Obtain the tolerances for the inequality constraints and the indices
            # for the violated constraints
            eps = getEps(
                dC,
                problem.n_eqconstraints,
                params["dt"],
                params["K"],
                norm_type=params["normalisation_norm"],
            )
            tildeEps = getTilde(C, problem.n_eqconstraints, eps=eps)
            print(f"eps: {eps}")
            # Obtain the violated contraints
            tilde = getTilde(C, problem.n_eqconstraints)

            p_matrix = p_matrix_eval(dC, tildeEps)
            q_vector = q_vector_eval(dJ, dC, tildeEps)
            qp_results = solve_dual_problem(
                p_matrix,
                q_vector,
                tildeEps,
                problem.n_eqconstraints,
                show_progress_qp=params["show_progress_qp"],
                tol_qp=params["tol_qp"],
            )
            muls = np.zeros(len(C))
            oldmuls = np.zeros(len(C))
            hat = np.asarray([False] * len(C))

            if qp_results:
                muls[tildeEps] = np.asarray(qp_results["x"]).flatten()
                oldmuls = muls.copy()
                hat = np.asarray([True] * len(C))
                hat[problem.n_eqconstraints :] = (
                    muls[problem.n_eqconstraints :] > 30 * params["tol_qp"]
                )
                if params.get("disable_dual", False):
                    hat = tildeEps

                dCdCT = dCdCT_eval(dC, hat)
                dCdCTinv = invert_dCdCT(dCdCT, params["debug"])
                muls = np.zeros(len(C))

                dCdJ = dCdJ_eval(dJ, dC, hat)
                muls[hat] = -dCdCTinv.dot(dCdJ[hat])

                if not np.all(muls[problem.n_eqconstraints :] >= 0):
                    display(
                        "Warning, the active set has not been predicted "
                        + "correctly Using old lagrange multipliers",
                        params["debug"],
                        level=1,
                        color="orange_4a",
                    )
                    hat = np.asarray([True] * len(C))
                    muls = oldmuls.copy()

            results["muls"].append(muls)
            display(
                f"Lagrange multipliers: {muls[:10]}", params["debug"], level=5
            )
            xiJ = xiJ_eval(dJ, dC, muls, hat)

            # Set of constraints union of active and new violated constraints.
            indicesEps = np.logical_or(tilde, hat)
            dCdCT = dCdCT_eval(dC, indicesEps)
            dCtdCtTinv = invert_dCdCT(dCdCT, params["debug"])

            xiC = xiC_eval(C, dC, dCtdCtTinv, alphas, indicesEps)

            # TODO Consider this? AC = min(0.9, alphaC * dt / max(compute_norm(xiC), 1e-9))
            AJ = params["alphaJ"]
            AC = params["alphaC"]

            # Make updates with merit function
            if xiC:
                problem.delta_x.assign(
                    Constant(-AJ) * xiJ - Constant(AC) * xiC
                )
            else:
                problem.delta_x.assign(Constant(-AJ) * xiJ)
            normdx = fd.norm(problem.delta_x)

            merit_eval_new = partial(merit_eval, muls, indicesEps, dCtdCtTinv)
            merit = merit_eval_new(AJ, J, AC, C)
            results["merit"].append(merit)
            if len(results["merit"]) > 3:
                print(
                    f"Merit oscillation: {(results['merit'][-1] - results['merit'][-2]) * (results['merit'][-2] - results['merit'][-3])}"
                )

            if descent_output_dir:
                descent_pvd.write(problem.delta_x)

            orig_phi.assign(phi)
            new_phi, newJ, newG, newH = line_search(
                problem,
                orig_phi,
                new_phi,
                merit_eval_new,
                merit,
                AJ,
                AC,
                dt=params["dt"],
                maxtrials=params["maxtrials"],
                tol_merit=params["tol_merit"],
                debug=params["debug"],
            )
            phi.assign(new_phi)
            (J, G, H) = (newJ, newG, newH)

        if params["monitor_time"]:
            print(
                f"Max time per iteration: {timings.max_time}, min time per iteration: {timings.min_time}"
            )

    results["J"].append(J)
    results["G"].append(G)
    results["H"].append(H)

    display("\n", params["debug"], -1)
    display("Optimization completed.", params["debug"], -1)
    return results
