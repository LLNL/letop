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
import scipy.sparse as sp


class Optimizable:
    """
    An abstract class whose instance can be passed as an input to
    nlspace_solve. An Optimizable object represents for describing a
    constrained minimization problem of the form

        min      J(x)
       x in X

       under the constraints
        g_i(x)=0  for all i=0..p-1
        h_i(x)<=0 for all i=0..q-1

    Attributes
    ----------
    nconstraints     : number of equality constraints g_i
    nineqconstraints : number of inequality constraints h_i
    h_size           : characteristic length scale used by the optimizer
                       to normalize descent directions and compute accuracy
                       bounds.
    is_manifold      : indicate whether the optimization domain X is a manifold
                       (in that case the inner product matrix shall be
                       recomputed and refactorized at every evaluation of a new
                       iterate x) or a Euclidean space.
    """

    def __init__(self):
        self.__currentJ_x = None
        self.__currentdJ_x = None
        self.nconstraints = 0
        self.nineqconstraints = 0
        self.is_manifold = True
        self.h_size = 1e-2

    def x0(self):
        """Returns the initialization."""
        return None

    def J(self, x):
        """Compute the value of the objective function at the point x."""
        return None

    def G(self, x):
        """Returns a list [g1,g2,...,gp] of the equality constraint
        values at x"""
        return []

    def H(self, x):
        """Returns a list [h1,h2,...,hp] of the inequality constraint
        values at x"""
        return []

    def dJ(self, x):
        """Returns the value of the Fréchet derivative of J at x.
        This must be a list or numpy array."""
        return None

    def dG(self, x):
        """Returns a list of the Fréchet derivatives of each of
            the equality constraints at x:
                [ [dg1dx1, dg1dx2,...,dg1dxk],
                  [dg2dx1,dg2dx2,...,dg2dxk],
                    ...
                  [dgpdx1, dgpdx2,...,dgpdxk] ]
        """
        return []

    def dH(self, x):
        """Returns
            a list of the Fréchet derivatives of each of the equality
            constraints at x:
                [ [dh1dx1, dh1dx2,...,dh1dxk],
                  [dh2dx1,dh2dx2,...,dh2dxk],
                    ...
                  [dhqdx1, dhqdx2,...,dhqdxk] ]
        """
        return []

    def dJT(self, x):
        """Returns the gradient of J at x.
        This must be implemented if no inner product is given.
        dJT(x) should return the solution of

                A * dJT(x) = dJ(x)

        where A is the chosen inner product.
        In particular, dJ(dJT(x)) should always be positive.
        """
        return None

    def dGT(self, x):
        """Returns the transpose of dG with respect to some inner product.
        This must be implemented if no inner product is given.
        dGT(x) should return a column matrix [ dG1T dG2T ... dGpT ]
        where each vector dGiT is the solution to

            A * dGiT(x) = dGi(x)

        where A is the chosen inner product.
        """
        return []

    def dHT(self, x):
        """Returns the transpose of dH with respect to some inner product.
        This must be implemented if no inner product is given.
        dHT(x) should return a column matrix [ dH1T dH2T ... dHpT ]
        where each vector dHiT is the solution to

            A * dHiT(x) = dHi(x)

        where A is the chosen inner product.
        """
        return []

    def eval(self, x):
        """Returns the triplet (J(x),G(x),H(x))"""
        return (self.J(x), self.G(x), self.H(x))

    def eval_sensitivities(self, x):
        """Returns the triplet (dJ(x),dG(x),dH(x))"""
        dJ = self.dJ(x)
        if self.nconstraints == 0:
            dG = np.empty((0, len(dJ)))
        else:
            dG = self.dG(x)
        if self.nineqconstraints == 0:
            dH = np.empty((0, len(dJ)))
        else:
            dH = self.dH(x)
        return (dJ, dG, dH)

    def eval_gradients(self, x):
        """Returns the triplet (dJT(x),dGT(x),dHT(x))
        Is used by nslpace_solve method only if self.inner_product returns
        None"""
        dJT = self.dJT(x)
        if self.nconstraints == 0:
            dGT = np.empty((0, len(dJT))).T
        else:
            dGT = self.dGT(x)
        if self.nineqconstraints == 0:
            dHT = np.empty((0, len(dJT))).T
        else:
            dHT = self.dHT(x)
        return (dJT, dGT, dHT)

    def retract(self, x, dx):
        """
        The retraction that explicit how to move from `x` by a step `dx`
        to obtain the new optimization point.

        Inputs :
            x  : current point
            dx : step (a vector of length len(dJ(x)))

        Output : the new optimized point after a step dx.
            newx = retract(x, dx)
        """
        return None

    def accept(self, results):
        """
        This function is called by nlspace_solve:
            - at the initialization
            - every time a new guess x is accepted on the optimization
              trajectory
        This allows to perform some post processing operations along the
        optimization trajectory. The function does not return any output but
        may update the dictionary `results` which may affect the optimization.
        Notably, the current point is stored in
            results['x'][-1]
        and an update of its value will be taken into account by nlspace_solve.

        Inputs:
            `results` : the current dictionary of results (see the function
                nlspace_solve)
        """
        pass


class EuclideanOptimizable(Optimizable):
    """An optimizable class for optimization in Euclidean space
    of dimension `n`.

    Usage
    -----
    >>> optim = EuclideanOptimizable(n);

    where n is the dimension of the space.
    The inner product matrix is automatically set to the identity of size `n`
    The retraction is the usual translation:
         x_{n+1} = x_n + dx

    Attribute:
        `n`  : The dimension of the Euclidean space.
    """

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.is_manifold = False

    def retract(self, x, dx):
        return x+dx



def checkOptimizable(problem: EuclideanOptimizable, x, eps=1e-6):
    """
    Check the implementation of derivatives by using finite differences

    Input
    -----

    problem : an instance of the EuclideanOptimizable class
    x       : a guess vector (dimension problem.n)
    eps     : increment for evaluating the finite difference

    """
    dx = np.random.rand(problem.n)
    (J, G, H) = problem.eval(x)
    (newJ, newG, newH) = tuple(map(
        np.asarray, problem.eval(np.asarray(x)+eps*dx)))
    (dJ, dG, dH) = tuple(map(np.asarray, problem.eval_sensitivities(x)))
    (checkJ, checkG, checkH) = ((newJ-J)/eps, (newG-G)/eps, (newH-H)/eps)
    (compareJ, compareG, compareH) = (dJ.dot(dx), dG.dot(dx), dH.dot(dx))
    print("Numerical sensitivities:")
    print(f"dJ.dx : \t expected {compareJ} vs. obtained {checkJ}")
    print(f"dG.dx : \t expected \n {compareG} \n vs. obtained \n {checkG}")
    print(f"dH.dx : \t expected \n {compareH} \n vs. obtained \n {checkH}")
