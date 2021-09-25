from lestofire_examples import (
    compliance_optimization,
    heat_exchanger_optimization,
)
from numpy.testing import assert_allclose


def test_heat_exchanger():
    results = heat_exchanger_optimization(n_iters=10)
    cost_func = results["J"][-1]
    assert_allclose(
        cost_func,
        -1.316,
        rtol=1e-2,
    )


def test_compliance():
    results = compliance_optimization(n_iters=10)
    cost_func = results["J"][-1]
    assert_allclose(
        cost_func,
        11.834513,
        rtol=1e-2,
    )
