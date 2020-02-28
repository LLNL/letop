from firedrake import assemble, Constant

from ufl import max_value

def h(g, lmbda, c):
    return max_value(g, -lmbda/c)

def augmented_lagrangian(*args, **kwargs):
    r'''
    Implements the augmented lagrangian term for the pointwise inequality constraint g < 0
    (where g is an integrand)
        \int_{dx} lmbda h + c * h * h
    where
        h = max(g, lmbda / c)
    '''

    g = args[0]
    lmbda = args[1]
    dx = args[2]
    c = args[3]

    value = assemble(lmbda * h(g, lmbda, c)*dx + c / Constant(2.0) * h(g, lmbda, c) * h(g, lmbda, c)*dx)

    return value
