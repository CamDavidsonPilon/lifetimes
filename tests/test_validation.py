import pytest
from lifetimes.data_compression import compress_bgext_data, compress_data, compress_session_session_before_conversion_data
from lifetimes.validation import generate_neg_likelihoods, goodness_of_test
import lifetimes.generate_data as gen
import pandas as pd
import lifetimes.estimation as est
from lifetimes.utils import multinomial_sample

sample_T = [2] * 100 + [3] * 100 + [4] * 100 + [5] * 100 + [6] * 100 + [7] * 100

@pytest.mark.validation
def test_generate_BG_neg_likelihoods():
    params = {'alpha': 0.32, 'beta': 0.85}

    simulation_size = 100
    N_users = 1000

    gen_data = compress_bgext_data(gen.bgext_model(T=sample_T,
                                                   alpha=params['alpha'],
                                                   beta=params['beta']))
    fitter = est.BGFitter(0.1)
    fitter.fit(**gen_data)
    n_lls = generate_neg_likelihoods(fitter=fitter, size=N_users, simulation_size=simulation_size)

    assert len(n_lls) == simulation_size
    assert n_lls.std() > 0


@pytest.mark.validation
def test_goodness_of_test_BG():
    params = {'alpha': 0.32, 'beta': 0.85}

    gen_data = compress_bgext_data(gen.bgext_model(T=sample_T,
                                                   alpha=params['alpha'],
                                                   beta=params['beta']))
    assert goodness_of_test(gen_data, fitter_class=est.BGFitter, verbose=True)

    # clearly not BG
    frequency = [0, 1, 2, 3, 4, 5]
    T = [5, 5, 5, 5, 5, 5]
    N = [10, 13, 17, 30, 40, 40]

    non_bg_data = pd.DataFrame({'frequency': frequency, 'T': T, 'N': N})

    assert not goodness_of_test(non_bg_data, fitter_class=est.BGFitter, verbose=True)

    # borderline BG
    frequency = [0, 1, 2, 3, 4, 5]
    T = [5, 5, 5, 5, 5, 5]
    N = [10, 10, 10, 30, 10, 40]

    borderline_bg_data = pd.DataFrame({'frequency': frequency, 'T': T, 'N': N})

    assert not goodness_of_test(borderline_bg_data, fitter_class=est.BGFitter, verbose=True)


@pytest.mark.validation
def test_goodness_of_test_BGBB():
    params = {'alpha': 0.32, 'beta': 0.85, 'gamma': 5, 'delta': 3}

    gen_data = compress_data(gen.bgbb_model(T=sample_T, size=100, **params))
    test_n = multinomial_sample(gen_data['N'])
    test_data = gen_data.copy(deep=True)
    test_data['N'] = test_n
    assert goodness_of_test(gen_data,
                            fitter_class=est.BGBBFitter,
                            verbose=True, test_data=test_data)


@pytest.mark.validation
def test_goodness_of_test_BGBBBG():
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    gen_data = gen.bgbbbg_model(T=sample_T, size=1000, compressed=True, **params)
    test_n = multinomial_sample(gen_data['N'])
    test_data = gen_data.copy(deep=True)
    test_data['N'] = test_n
    assert goodness_of_test(gen_data,
                            fitter_class=est.BGBBBGFitter,
                            verbose=True, test_data=test_data)

