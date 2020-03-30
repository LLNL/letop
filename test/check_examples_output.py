import sys
import re
from collections import namedtuple
from numpy.testing import assert_allclose

TestExample = namedtuple('TestExample', ['folder', 'n_iter', 'value'])


test_collection = [TestExample(folder='heat_exchanger', n_iter=88, value=-5783.83288)]

def run_check(output_file, folder):

    for test in test_collection:
        if folder == test.folder:
            current_test = test
            break

    if current_test is None:
        raise ValueError('No current test data for example {} exists'.format(folder))

    with open(output_file, 'r') as read_test:
        all_iters = re.findall('It: ([0-9]*) Obj: (-?[0-9]*.[0-9]*)', read_test.read())
        last_iter = all_iters[-1]
        assert current_test.n_iter == int(last_iter[0]), 'Number of iterations for {} is not matching'.format(current_test.folder)
        assert_allclose(float(last_iter[1]), current_test.value, rtol=1e-4, err_msg='Cost function for {} is not matching'.format(current_test.folder))


if __name__=="__main__":
    assert len(sys.argv) > 2, "No output file and/or example folder was provided"
    folder = sys.argv[2]
    output_file = sys.argv[1]
    run_check(output_file, folder)
