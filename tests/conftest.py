from __future__ import print_function
import numpy as np


def pytest_runtest_setup(item):
    seed = 868
    print("Seed used in np.random.seed(): %d" % seed)
    np.random.seed(seed)
