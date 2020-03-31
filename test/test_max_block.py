from lestofire import max_fvalue, augmented_lagrangian_float

from firedrake import  dx, FunctionSpace, UnitSquareMesh, Constant, Function, \
                        assemble, UnitSquareMesh

from pyadjoint import AdjFloat, ReducedFunctional, Control, get_working_tape
from pyadjoint.tape import Tape, set_working_tape


def test_max_fvalue():
    a = AdjFloat(1.0)
    b = AdjFloat(2.0)
    c = max_fvalue(a, b)
    rfa = ReducedFunctional(c, Control(a))
    assert rfa(a) == 2.0

    # a > b
    assert rfa(AdjFloat(4.0)) == 4.0
    assert rfa.derivative() == 1.0

    assert rfa(AdjFloat(1.0)) == 2.0
    assert rfa.derivative() == 0.0


    tape = Tape()
    set_working_tape(tape)

    a = AdjFloat(1.0)
    b = AdjFloat(5.0)
    c = max_fvalue(a, b)
    # a < b
    rfb = ReducedFunctional(c, Control(b))
    assert rfb(b) == 5.0
    assert rfb(AdjFloat(0.5)) == 1.0
    assert rfb.derivative() == 0.0

def test_augmented_lagrangian_float():

    mesh = UnitSquareMesh(10, 10)

    a = Constant(3.0)
    G = assemble(Constant(5.0)*a*dx(domain=mesh))
    lmbda = AdjFloat(1.0)
    c = AdjFloat(40.0)
    J = augmented_lagrangian_float(G, lmbda, c)

    rf = ReducedFunctional(J, Control(a))
    exact_value = lmbda*5.0 + c*75.0
    from numpy.testing import assert_allclose
    assert_allclose(rf.derivative().values()[0], exact_value, rtol=1e-8)

    # Now test the other branch of the conditional, that has zero derivative
    tape = Tape()
    set_working_tape(tape)

    a = Constant(3.0)
    G = assemble(Constant(5.0)*a*dx(domain=mesh))
    lmbda = AdjFloat(-1.0)
    c = AdjFloat(0.01)
    J = augmented_lagrangian_float(G, lmbda, c)
    rf = ReducedFunctional(J, Control(a))
    exact_value = 0.0
    assert_allclose(rf.derivative().values()[0], exact_value, rtol=1e-8)
