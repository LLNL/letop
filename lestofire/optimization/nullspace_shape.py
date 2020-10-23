# Copyright 2018-2019 CNRS, Ecole Polytechnique and Safran.
#
# This file is part of nullspace_optimizer.
#
# nullspace_optimizer is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# nullspace_optimizer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# A copy of the GNU General Public License is included below.
# For further information, see <http://www.gnu.org/licenses/>.

import numpy as np
import cvxopt as cvx
from .interface import InfDimProblem
import scipy.sparse.linalg as linalg
import scipy.sparse as sp
import time

from firedrake import norm, assemble, inner, Function, dx, Constant, PETSc
from pyadjoint import no_annotations


@no_annotations
def nlspace_solve(problem: InfDimProblem, params=None, results=None, delta_x_pvd = None, delta_x_func = None):
    """
    Solve the optimization problem
        min      J(x)
        x in V
        under the constraints
        g_i(x)=0  for all i=0..p-1
        h_i(x)<=0 for all i=0..q-1

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
              the dictionary `results['x'][-1]`.
              Useful to restart an optimization after an interruption.

    Output
    ------
    results : dictionary containing
        results['x']       : optimization path
                             (x_0,x_1,...,x_n).
        results['J']       : values of the objective function along the path
                             (J(x_0),...,J(x_n))
        results['G']       : equality constraint values
                             (G(x_0),...,G(x_n))
        results['H']       : inequality constraints values
                             (H(x_0),...,H(x_n))
        results['muls']    : lagrange multiplier values
                             (mu(x_0),...,mu(x_n))
        results['normxiJ'] : norms of the nullspace step xiJ
        results['tolerance']:estimation of an uncertainty bound on the
                             constraints under which these can expect to be
                             satisfied. It is computed thanks to the formula:
                                tolerance = ||DC||_1 dt
        results['s']       : the optimization path length
                             (s(x_0),s(x_1),...,s(x_n))
                             with s(x(t))=\\int_0^T ||x'(t)|| dt


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

    params['dt']       : (default : `problem.h_size`). Pseudo time-step
        expressed in a length unit. The descent step is normalized such that
            ||dxiJ||=alphaJ*dt ||dxiC||=alphaC*dt
        for the first params['itnormalisation'] iterations.  If `dt` is not
        set, then the update `dt`<-problem.h_size is made at every iteration
        (useful if problem.h_size depends on the current iterate x)

    params['itnormalisation']: (default 1) the iteration number after which the
        null space step xiJ is not normalized anymore

    params['K']: tunes the distance at which inactive inequality constraints
        are felt. Constraints are felt from a distance K*params['dt']

    params['maxit']    : Maximal number of iterations (default : 4000)

    params['maxtrials']: (default 3) number of trials in between time steps
        until the merit function decreases

    params['normalisation_norm'] : the euclidean norm used to normalize the
        descent direction.

    params['normalize_tol'] : if >= 0 (default : 0),
        then xiJ is normalized every time the set of active constraints changes
        in addition to the first params['itnormalisation'] iterations.  The
        value of this parameter can be set to a strictly positive tolerance
        (e.g. 1e-7) to normalize only when a substantial discontinuity occurs
        in the multipliers. Useful if the optimizer is hesitating between
        active constraints If set to a negative value, then no normalization is
        performed when the active set changes.

    params['provide_gradient']   : (default False).
        If set to True, then the algorithm will call problem.dJ, problem.dGT,
        problem.dHT to compute gradients

    params['tol']      : (default 1e-7) Algorithm stops when
            ||x_{n+1}-x_n||<params['tol']
        or after params['maxit'] iterations.

    params['tol_merit'] : (default 0) a new iterate x_{n+1} is accepted if
        merit(x_{n+1})<(1+sign(merit(x_n)*params['tol_merit']))*merit(x_n)

    params['tol_qp'] : (default 1e-20) the tolerance for the qp solver cvxopt

    params['show_progress_qp'] : (default False) If true, then the output of
        cvxopt will be displayed between iterations.

    params['inner_prod_solver'] : 'umfpack' (default) or 'cg'. The solver used
        to invert the inner product when computing gradients.
    """
    if params is None:
        params = dict()
    alphaJ = params.get("alphaJ", 1)
    alphaC = params.get("alphaC", 1)
    maxit = params.get("maxit", 4000)
    maxtrials = params.get("maxtrials", 3)
    debug = params.get("debug", 0)
    normalisation_norm = params.get("normalisation_norm", np.inf)
    itnormalisation = params.get("itnormalisation", 1)
    tol_merit = params.get("tol_merit", 0)
    inner_prod_solver = params.get("inner_prod_solver", "umfpack")
    dt = params.get("dt", problem.h_size)
    K = params.get("K", 0.1)
    provide_gradient = params.get("provide_gradient", False)
    tol_qp = params.get("tol_qp", 1e-20)
    show_progress_qp = params.get("show_progress_qp", False)
    normalize_tol = params.get("normalize_tol", 0)

    alphas = np.asarray(
        params.get(
            "alphas", [1] * (len(problem.eqconstraints) + len(problem.ineqconstraints))
        )
    )

    p = len(problem.eqconstraints)
    q = len(problem.ineqconstraints)

    if hasattr(problem, "inner_product"):
        norm_func = lambda x: np.sqrt(problem.inner_product(x, x))
        inner_product = problem.inner_product
    else:
        norm_func = norm
        inner_product = lambda x, y: assemble(inner(x, y) * dx, annotate=False)

    try:
        import colored as col

        def display(message, level=0, color=None):
            if color:
                message = col.stylize(message, col.fg(color))
            if debug >= level:
                print(message)

    except Exception:

        def display(message, level=0, color=None):
            if debug >= level:
                print(message)

    def compute_norm(f):
        if normalisation_norm == np.inf:
            with f.dat.vec_ro as vu:
                return vu.norm(PETSc.NormType.NORM_INFINITY)
        elif normalisation_norm == 2:
            normx = norm_func(f)
            return normx

    def getTilde(C, eps=0):
        """
        Obtain the set of violated inequality constraints
        """
        tildeEps = C[p:] >= -eps
        tildeEps = np.asarray(np.concatenate(([True] * p, tildeEps)), dtype=bool)
        return tildeEps

    def getEps(C, dC):
        if len(dC) == 0:
            return (0, [])
        if normalisation_norm == np.inf:
            eps = []
            for dCi in dC[p:]:
                with dCi.dat.vec_ro as dCv:
                    norm_inf = dCv.norm(PETSc.NormType.NORM_INFINITY)
                    eps.append(norm_inf)
            eps = np.array(eps) * dt * K
        elif normalisation_norm == 2:
            eps = np.array([norm_func(dCi) * dt * K for dCi in dC[p:]])
        tildeEps = getTilde(C, eps)
        return (eps, tildeEps)

    def checkResults(results):
        """ Check the output dictionary in order to restart
            from the last iterate"""
        n = results["x"].function_space().dim()
        group1 = ["J", "G", "H", "s"]
        group2 = ["normxiJ", "eps", "muls", "tolerance"]
        for key in group1:
            if key not in results:
                raise Exception(
                    f"Error, key {key} is missing in the results dictionary."
                )
            if len(results[key]) != n:
                raise Exception(f"Error, key {key} should have length n={n}.")
        for key in group2:
            if key not in results:
                raise Exception(
                    f"Error, key {key} is missing in the results dictionary."
                )
            if len(results[key]) == n:
                results[key] = results[key][:-1]
            if len(results[key]) != n - 1:
                raise Exception(f"Error, key {key} should have length n-1={n-1}.")

    if results and "J" in results and len(results["J"]) > 0:
        raise Exception("Restarting capabilities not implemented yet")
        ## Allow to restart the optimization from the last result
        # checkResults(results)
        # x = results["x"][-1]
        # for key in results.keys():
        #    if key not in ["normxiJ", "eps", "muls", "s", "tolerance"]:
        #        results[key] = results[key][:-1]
        # if "muls" in results:
        #    results["muls"] = [np.asarray(muls) for muls in results["muls"]]
        #    for i in reversed(range(len(results["muls"]))):
        #        if i < itnormalisation:
        #            break
        #        if normalize_tol >= 0 and not np.all(
        #            (results["muls"][i][p:] > normalize_tol)
        #            == (results["muls"][i - 1][p:] > normalize_tol)
        #        ):
        #            break
        #    itnormalisation = max(itnormalisation, i + 1)
        #    display(
        #        f"Last normalisation index found: "
        #        + f"itnormalisation={itnormalisation}",
        #        level=5,
        #        color="magenta",
        #    )
    else:
        results = dict()
        results["x"] = []
        results["J"] = []
        results["G"] = []
        results["H"] = []
        results["muls"] = []
        results["normxiJ"] = []
        results["tolerance"] = []
        results["s"] = [0]  # Longueur du chemin d'optimization parcouru
        results["eps"] = []

        x = problem.x0()

    (J, G, H) = problem.eval(x)

    tol = params.get("tol", 1e-5 * dt)
    normdx = 1  # current value for x_{n+1}-x_n

    while normdx > tol and len(results["J"]) < maxit:
        results["J"].append(J)
        results["G"].append(G)
        results["H"].append(H)
        results["x"].append(x)
        problem.accept(results)

        if hasattr(problem, "reinit"):
            problem.reinit(x)

        it = len(results["J"]) - 1
        display("\n", 1)
        display(
            f"{it}. J="
            + format(J, ".4g")
            + " "
            + "G=["
            + ",".join([format(x, ".4g") for x in G[:10]])
            + "] "
            + "H=["
            + ",".join([format(x, ".4g") for x in H[:10]])
            + "] "
            + " ||dx||_V="
            + format(normdx, ".4g"),
            0,
        )

        x = results["x"][-1]
        display(f"x={x}", level=5)

        # Returns the gradients (in the primal space). They are
        # firedrake.Function's
        (dJ, dG, dH) = problem.eval_gradients(x)

        H = np.asarray(H)
        G = np.asarray(G)
        C = np.concatenate((G, H))

        dC = dG + dH

        dt = params.get("dt", problem.h_size)
        # Obtain the tolerances for the inequality constraints and the indices
        # for the violated constraints
        (eps, tildeEps) = getEps(C, dC)
        print(f"eps: {eps}")
        results["eps"].append(eps)
        # Obtain the violated contraints
        tilde = getTilde(C)
        qtildeEps = sum(tildeEps) - p

        # Solve the dual problem to obtain the new set of active constraints
        # Solve it using a quadratic solver:
        # minimize    (1/2)*x'*P*x + q'*x
        # subject to  G*x <= h
        #            A*x = b.
        # A and b are optionals and not given here.
        # sum(tildeEps) = p + qtildeEps
        Pmatrix = np.zeros((sum(tildeEps), sum(tildeEps)))
        qvector = np.zeros((sum(tildeEps), 1))

        ii = 0
        for i, tildei in enumerate(tildeEps):
            jj = 0
            for j, tildej in enumerate(tildeEps):
                if tildej and tildei:
                    # Pmatrix[ii, jj] = assemble(inner(dC[i], dC[j])*dx)
                    Pmatrix[ii, jj] = inner_product(dC[i], dC[j])
                    jj += 1
            if tildei:
                # qvector[ii] = assemble(inner(dJ, dC[i])*dx)
                qvector[ii] = inner_product(dJ, dC[i])
                ii += 1

        Pcvx = cvx.matrix(Pmatrix)
        qcvx = cvx.matrix(qvector)
        Gcvx = cvx.matrix(
            np.concatenate((np.zeros((qtildeEps, p)), -np.eye(qtildeEps)), axis=1)
        )
        hcvx = cvx.matrix(np.zeros((qtildeEps, 1)))
        muls = np.zeros(len(C))
        oldmuls = np.zeros(len(C))
        hat = np.asarray([False] * len(C))
        if p + qtildeEps > 0:
            ret = cvx.solvers.qp(
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
            muls[tildeEps] = np.asarray(ret["x"]).flatten()
            oldmuls = muls.copy()
            hat = np.asarray([True] * len(C))
            hat[p:] = muls[p:] > 10 * tol_qp
            if params.get("disable_dual", False):
                hat = tildeEps

            # Compute null space direction xiJ
            dCdCT = np.zeros((sum(hat), sum(hat)))
            ii = 0
            for i, tildei in enumerate(hat):
                jj = 0
                for j, tildej in enumerate(hat):
                    if tildej and tildei:
                        # dCdCT[ii, jj] = assemble(inner(dC[i], dC[j])*dx)
                        dCdCT[ii, jj] = inner_product(dC[i], dC[j])
                        jj += 1
                if tildei:
                    ii += 1

            try:
                dCdCTinv = np.linalg.inv(dCdCT)
            except Exception:
                display(
                    "Warning, constraints are not qualified, using " "pseudo-inverse.",
                    -1,
                    color="red",
                )
                dCdCTinv = np.linalg.pinv(dCdCT)
            muls = np.zeros(len(C))
            dCdJ = muls.copy()
            ii = 0
            for i, hati in enumerate(hat):
                if hati:
                    # dCdJ[ii] = assemble(inner(dC[i], dJ)*dx)
                    dCdJ[ii] = inner_product(dC[i], dJ)
                    ii += 1
            muls[hat] = -dCdCTinv.dot(dCdJ[hat])

            if not np.all(muls[p:] >= 0):
                display(
                    "Warning, the active set has not been predicted "
                    + "correctly Using old lagrange multipliers",
                    level=1,
                    color="orange_4a",
                )
                hat = np.asarray([True] * len(C))
                muls = oldmuls.copy()

        results["muls"].append(muls)
        display(f"Lagrange multipliers: {muls[:10]}", level=5)
        # This is a sum of the functions in Firedrake
        xiJ = Function(problem.fespace())
        list_func = []
        if hat.any():
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

        # Compute range step direction xiC
        indicesEps = np.logical_or(tilde, tildeEps)
        dCdCT = np.zeros((sum(indicesEps), sum(indicesEps)))
        ii = 0
        for i, indi in enumerate(indicesEps):
            jj = 0
            for j, indj in enumerate(indicesEps):
                if indi and indj:
                    # dCdCT[ii, jj] = assemble(inner(dC[i], dC[j])*dx)
                    dCdCT[ii, jj] = inner_product(dC[i], dC[j])
                    jj += 1
            if indi:
                ii += 1

        try:
            dCtdCtTinv = np.linalg.inv(dCdCT)
        except Exception:
            display(
                "Warning, constraints are not qualified. " + "Using pseudo-inverse.",
                1,
                color="orange_4a",
            )
            dCtdCtTinv = np.linalg.pinv(dCdCT)
        # Sum of functions over indicesEps
        xiC = Function(problem.fespace())
        if indicesEps.any():
            list_func = []
            dCdTinvCalpha = dCtdCtTinv.dot(C[indicesEps] * alphas[indicesEps])
            ii = 0
            for i, indi in enumerate(indicesEps):
                if indi:
                    list_func.append(Constant(dCdTinvCalpha[ii]) * dC[i])
                    ii += 1
            xiC.assign(sum(list_func))

        results["normxiJ"].append(compute_norm(xiJ))
        if len(results["J"]) <= itnormalisation or (
            normalize_tol >= 0
            and not np.all(
                (results["muls"][-1][p:] > normalize_tol)
                == (results["muls"][-2][p:] > normalize_tol)
            )
        ):
            AJ = alphaJ * dt / (1e-9 + results["normxiJ"][-1])
            itnormalisation = max(itnormalisation, len(results["J"]))
            display(
                f"Normalisation of the xiJ direction, "
                + f"itnormalisation={itnormalisation}",
                level=5,
            )
        else:
            AJ = (
                alphaJ
                * dt
                / max(
                    1e-9 + results["normxiJ"][-1],
                    results["normxiJ"][itnormalisation - 1],
                )
            )
        AC = min(0.9, alphaC * dt / max(compute_norm(xiC), 1e-9))

        print("AJ : {0:.5f}, AC: {1:.5f}".format(AJ, AC))

        # Make updates with merit function
        delta_x = Function(problem.fespace())
        delta_x.assign(Constant(-AJ) * xiJ - Constant(AC) * xiC)
        normdx = norm_func(delta_x)
        if delta_x_pvd:
            delta_x_func.assign(delta_x)
            delta_x_pvd.write(delta_x_func)
        success = 0

        results["tolerance"].append(np.sum([norm(dCi, "L1") for dCi in dC]) * dt)

        merit = AJ * (J + muls.dot(C)) + 0.5 * AC * C[indicesEps].dot(
            dCtdCtTinv.dot(C[indicesEps])
        )
        for k in range(maxtrials):
            # Is x modified by retract here?
            newx = problem.retract(x, delta_x, scaling=(0.5 ** k))
            (newJ, newG, newH) = problem.eval(newx)
            newC = np.concatenate((newG, newH))
            newmerit = AJ * (newJ + muls.dot(newC)) + 0.5 * AC * newC[indicesEps].dot(
                dCtdCtTinv.dot(newC[indicesEps])
            )
            print(f"newJ={newJ}, newC={newC}")
            print(f"merit={merit}, newmerit={newmerit}")
            if newmerit < (1 + np.sign(merit) * tol_merit) * merit:
                success = 1
                break
            else:
                display(
                    "Warning, merit function did not decrease "
                    + f"(merit={merit}, newmerit={newmerit})"
                    + f"-> Trial {k+1}",
                    0,
                    "red",
                )
        if not success:
            display(
                "All trials have failed, passing to the next iteration.", color="red"
            )
            problem.restore()
        else:
            x = newx
        (J, G, H) = (newJ, newG, newH)

        with delta_x.dat.vec_ro as vu:
            delta_x_inf_norm = vu.norm(PETSc.NormType.NORM_INFINITY)
        results["s"].append(results["s"][-1] + delta_x_inf_norm)

    results["J"].append(J)
    results["G"].append(G)
    results["H"].append(H)
    results["x"].append(x)
    problem.accept(results)

    display("\n", -1)
    display("Optimization completed.", -1)
    display(
        f"{it+1}. J="
        + format(J, ".4g")
        + " "
        + "G=["
        + ",".join([format(x, ".4g") for x in G[:10]])
        + "] "
        + "H=["
        + ",".join([format(x, ".4g") for x in H[:10]])
        + "] "
        + " ||dx||_V="
        + format(normdx, ".4g"),
        -1,
        color="blue",
    )
    return results
