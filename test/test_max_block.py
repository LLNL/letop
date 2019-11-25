from lestofire import max_fvalue

from pyadjoint import AdjFloat, ReducedFunctional, Control, get_working_tape
from pyadjoint.tape import Tape, set_working_tape


#def test_max_fvalue():
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
