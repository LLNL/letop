from lestofire import max_fvalue

from pyadjoint import AdjFloat, ReducedFunctional, Control


def test_max_fvalue():
    a = AdjFloat(1.0)
    b = AdjFloat(2.0)
    c = max_fvalue(a, b)
    rf = ReducedFunctional(c, Control(a))
    assert rf(a) == 2.0
    assert rf(AdjFloat(4.0)) == 4.0
    assert rf.derivative() == 1.0
    assert rf(AdjFloat(1.0)) == 2.0
    assert rf.derivative() == 0.0
