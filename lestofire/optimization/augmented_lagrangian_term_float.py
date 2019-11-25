from firedrake_adjoint import assemble, Constant
from pyadjoint import Block

import numpy as np


def max_fvalue(a, b):
    return max(a, b)

backend_max_fvalue = max_fvalue

class MaxBlock(Block):

    """Docstring for MaxBlock. """

    def __init__(self, a, b, **kwargs):
        super(MaxBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(a)
        self.b = b

    def __str__(self):
        return "MaxBlock"

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_max_fvalue(inputs[0], self.b)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        a = inputs[idx]

        if a > self.b:
            return adj_input
        else:
            return 0.0


def h(g, lmbda, c):
    return max_fvalue(g, -lmbda/c)

def augmented_lagrangian(*args, **kwargs):
    r'''
    Implements the augmented lagrangian term for the inequality constraint g < 0
        lmbda h + c * h * h
    where
        h = max(g, lmbda / c)
    '''

    g = args[0]
    lmbda = args[1]
    c = args[3]

    value = lmbda * h(g, lmbda, c) + c / Constant(2.0) * h(g, lmbda, c) * h(g, lmbda, c)

    return value

from pyadjoint.overloaded_function import overload_function
max_fvalue = overload_function(max_fvalue, MaxBlock)

