import pytest
from lifetimes.data_compression import compress_bgext_data
from lifetimes.validation import generate_BG_neg_likelihoods, goodness_of_test_BG
import lifetimes.generate_data as gen
import matplotlib.pyplot as plt
import pandas as pd
import random


@pytest.mark.validation
def test_generate_BG_neg_likelihoods():
    params = {'alpha': 0.32, 'beta': 0.85}

    simulation_size = 100
    N_users = 1000
    T_horizon = 10
    n_lls = generate_BG_neg_likelihoods(params['alpha'], params['beta'], T=T_horizon, size=N_users,
                                        simulation_size=simulation_size)

    assert len(n_lls) == simulation_size
    assert n_lls.std() > 0


@pytest.mark.validation
def test_goodness_of_test_BG():
    params = {'alpha': 0.32, 'beta': 0.85}

    gen_data = compress_bgext_data(gen.bgext_model(T=[2] * 100 + [3] * 100
                                                     + [4] * 100 + [5] * 100
                                                     + [6] * 100 + [7] * 100,
                                                   alpha=params['alpha'],
                                                   beta=params['beta']))
    assert goodness_of_test_BG(gen_data, verbose=True)

    # clearly not BG
    frequency = [0, 1, 2, 3, 4, 5]
    T = [5, 5, 5, 5, 5, 5]
    N = [10, 13, 17, 30, 40, 40]

    non_bg_data = pd.DataFrame({'frequency': frequency, 'T': T, 'N': N})

    assert not goodness_of_test_BG(non_bg_data, verbose=True)

    # borderline BG
    frequency = [0, 1, 2, 3, 4, 5]
    T = [5, 5, 5, 5, 5, 5]
    N = [10, 10, 10, 30, 10, 40]

    borderline_bg_data = pd.DataFrame({'frequency': frequency, 'T': T, 'N': N})

    assert not goodness_of_test_BG(borderline_bg_data, verbose=True)

