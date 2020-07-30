import sys
import re
from collections import namedtuple
from numpy.testing import assert_allclose

TestExample = namedtuple("TestExample", ["folder", "n_iter", "cost_func_value"])


# Comparison of final iterations is pointless. These are highly nonlinear
# problems and a small difference can create a noticeably difference in the
# final number of iterations and the cost function value.
test_collection = [
    TestExample(folder="heat_exchanger", n_iter=10, cost_func_value=-0.2654),
    TestExample(folder="cantilever", n_iter=10, cost_func_value=1479),
    TestExample(folder="stokes", n_iter=10, cost_func_value=7.575e04),
]


def run_check(output_file, folder):

    current_test = None
    for test in test_collection:
        if folder == test.folder:
            current_test = test
            break

    if current_test is None:
        raise ValueError("No current test data for example {} exists".format(folder))

    with open(output_file, "r") as read_test:
        all_iters = re.findall("([0-9]*)\. J=(-?[0-9]*.[0-9]*)", read_test.read())
        last_iter = all_iters[10]
        assert current_test.n_iter == int(
            last_iter[0]
        ), "Number of iterations for {} is not matching".format(current_test.folder)
        assert_allclose(
            float(last_iter[1]),
            current_test.cost_func_value,
            rtol=1e-4,
            err_msg="Cost function for {} is not matching".format(current_test.folder),
        )


if __name__ == "__main__":
    assert len(sys.argv) > 2, "No output file and/or example folder was provided"
    folder = sys.argv[2]
    output_file = sys.argv[1]
    run_check(output_file, folder)
