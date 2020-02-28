from firedrake import assemble, Constant
from pyadjoint import Block, AdjFloat
from pyadjoint.tape import no_annotations

import numpy as np


def max_fvalue(a, b):
    return max(float(a), float(b))

backend_max_fvalue = max_fvalue

class MaxBlock(Block):

    """Docstring for MaxBlock. """

    def __init__(self, a, b, **kwargs):
        super(MaxBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(a)
        self.add_dependency(b)

    def __str__(self):
        return "MaxBlock"

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_max_fvalue(inputs[0], inputs[1])

    @no_annotations
    def evaluate_adj(self, markings=False):
        a = self.get_dependencies()[0].saved_output
        b = self.get_dependencies()[1].saved_output
        adj_input = self.get_outputs()[0].adj_value

        if adj_input is None:
            return

        if a > b:
            self.get_dependencies()[0].add_adj_output(adj_input)
            self.get_dependencies()[1].add_adj_output(0.0)
        else:
            self.get_dependencies()[0].add_adj_output(0.0)
            self.get_dependencies()[1].add_adj_output(adj_input)

from pyadjoint.overloaded_function import overload_function
max_fvalue = overload_function(max_fvalue, MaxBlock)

def h(g, lmbda, c):
    return max_fvalue(g, -lmbda/c)

def augmented_lagrangian_float(*args, **kwargs):
    r'''
    Implements the augmented lagrangian term for the inequality constraint g < 0
        lmbda h + c * h * h
    where
        h = max(g, lmbda / c)
    '''

    g = args[0]
    lmbda = args[1]
    c = args[2]

    value = lmbda * h(g, lmbda, c) + c / 2.0 * h(g, lmbda, c) * h(g, lmbda, c)

    return value


