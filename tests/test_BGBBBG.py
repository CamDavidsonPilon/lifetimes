from __future__ import print_function
import pytest
import math
import lifetimes.generate_data as gen
import numpy as np
import lifetimes.estimation as est
import lifetimes.models as mod
from lifetimes.data_compression import compress_data, compress_session_session_before_conversion_data
from lifetimes.data_compression import filter_data_by_T
from lifetimes import models
import pandas as pd
from lifetimes.utils import is_almost_equal, is_same_order
from uncertainties import correlation_matrix, ufloat


@pytest.mark.BGBBBB
def test_BGBBBG_generation():
    T = 100
    size = 100

    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    gen_data = gen.bgbbbg_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                                params['epsilon'],
                                params['zeta'], size=size)

    assert len(gen_data) == 100
    assert 'T' in gen_data
    assert 'frequency' in gen_data
    assert 'frequency_before_conversion' in gen_data
    assert 'recency' in gen_data
    assert 'p' in gen_data
    assert 'c' in gen_data
    assert 'theta' in gen_data
    assert 'alive' in gen_data


@pytest.mark.BGBBBG
def test_BGBBBG_generation_with_time_first_purchase():
    T = 100
    size = 10000

    params = {'alpha': 0.48, 'beta': 0.96, 'gamma': 0.25, 'delta': 1.3, 'epsilon': 1.0, 'zeta': 10.0}

    gen_data = gen.bgbbbg_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                                params['epsilon'],
                                params['zeta'], size=size, time_first_purchase=True)

    assert 'time_first_purchase' in gen_data

    # build conversion profile
    ts = range(T + 1)
    cs = [0] * len(ts)
    cs_err = [0] * len(ts)
    tfp_column = gen_data['time_first_purchase']
    N = len(tfp_column)
    for t in ts:
        n = float(len(tfp_column[tfp_column == t]))
        p = n / N
        cs[t] = p
        cs_err[t] = np.sqrt(1.0 / N * p * (1.0 - p))

    conversion_data_frame = pd.DataFrame({'t': ts, 'c': cs, 'c_err': cs_err})


@pytest.mark.BGBBBG
def test_likelyhood():
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    # test simgle number as inputs
    freq = 1
    sess_before_purch = 1
    rec = 1
    T = 2
    ll = est.BGBBBGFitter._negative_log_likelihood(params.values(), freq, rec, T, sess_before_purch, penalizer_coef=0,
                                                   N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test arrays as inputs
    freq = [1, 0]
    sess_before_purch = [1, 1]
    rec = [1, 0]
    T = [2, 2]
    ll = est.BGBBBGFitter._negative_log_likelihood(params.values(), freq, rec, T, sess_before_purch, penalizer_coef=0,
                                                   N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test np.arrays as inputs
    freq = np.array([1, 0])
    sess_before_purch = np.array([1, 1])
    rec = np.array([1, 0])
    T = np.array([2, 2])
    ll1 = est.BGBBBGFitter._negative_log_likelihood(params.values(), freq, rec, T, sess_before_purch, penalizer_coef=0,
                                                    N=None)

    assert ll == ll1
    assert isinstance(ll1, float) or isinstance(ll1, int)


@pytest.mark.BGBBBB
def test_BGBBBG_fitting_compressed_or_not():
    T = 30
    size = 10000

    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    data = gen.bgbbbg_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                            params['epsilon'],
                            params['zeta'], size=size, time_first_purchase=True)

    compressed_data = compress_session_session_before_conversion_data(data)

    # fitter = est.BGBBBGFitter()
    fitter_compressed = est.BGBBBGFitter()

    # fitter.fit(data['frequency'], data['recency'], data['T'], data['frequency_before_conversion'],
    #           initial_params=params.values())
    fitter_compressed.fit(compressed_data['frequency'], compressed_data['recency'], compressed_data['T'],
                          compressed_data['frequency_before_conversion'],
                          N=compressed_data['N'], initial_params=params.values())

    print(params)
    # print fitter.params_
    print(fitter_compressed.params_)
    tot = 0
    fitted_conv = []
    for t in range(31):
        res = fitter_compressed.expected_probability_of_converting_at_time(t)
        tot += res
        fitted_conv.append(res)
    print(tot)

    conversion_instants = data['time_first_purchase']
    real_conv = [0] * 31
    for t in conversion_instants:
        if not math.isnan(t):
            real_conv[int(t)] += 1.0
    real_conv = [x / len(conversion_instants) for x in real_conv]

    for r, f in zip(real_conv, fitted_conv):
        print(r, " - ", f)

    print(sum(real_conv), " - ", sum(fitted_conv))


    # for par_name in params.keys():
    # assert math.fabs(fitter.params_[par_name] - fitter_compressed.params_[par_name]) < 0.00001


@pytest.mark.BGBB
def test_BGBBBGExt_integration_in_models_with_uncertainties():
    T = 10
    size = 100

    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0, 'c0': 0.05}

    data = gen.bgbbbgext_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                               params['epsilon'],
                               params['zeta'], params['c0'], size=size, time_first_purchase=True)

    compressed_data = compress_session_session_before_conversion_data(data)

    model = mod.BGBBBGExtModel(penalizer_coef=0.2)

    model.fit(frequency=compressed_data['frequency'], recency=compressed_data['recency'], T=compressed_data['T'],
              frequency_before_conversion=compressed_data['frequency_before_conversion'],
              N=compressed_data['N'], initial_params=params.values())

    print("Generation params")
    print(params)

    print("Fitted params")
    print(model.params)
    print(model.params_C)

    print("Uncertain parameters")
    print(model.uparams)

    # test correlations preserved
    assert is_almost_equal(correlation_matrix([model.uparams['alpha'], model.uparams['alpha']])[0, 1], 1.0)
    assert 1.0 > correlation_matrix([model.uparams['alpha'] + ufloat(1, 1), model.uparams['alpha']])[0, 1] > 0.0

    # stub of profile
    p1 = model.expected_number_of_sessions_up_to_time(1)
    p2 = model.expected_number_of_sessions_up_to_time(2)

    assert 1.0 > correlation_matrix([p1, p2])[0, 1] > 0.0

    print("E[X(t)] as a function of t")
    for t in [0, 1, 10, 100, 1000, 10000]:
        uEx = model.expected_number_of_sessions_up_to_time(t)
        print(t, uEx)
        assert uEx.n >= 0
        assert uEx.s >= 0

    t = 10
    print("E[X(t) = n] as a function of n, t = " + str(t))
    tot_prob = 0.0
    for n in range(t + 1):
        prob = model.fitter.probability_of_n_sessions_up_to_time(t, n)
        print(n, prob)
        tot_prob += prob
        assert 1 >= prob >= 0

        uprob = model.probability_of_n_sessions_up_to_time(t, n)
        print(uprob)
        assert is_almost_equal(uprob.n, prob)

    assert math.fabs(tot_prob - 1.0) < 0.00001

    print("c(t) as a function of t")
    for t in [0, 1, 10, 100, 1000]:
        uc = model.expected_probability_of_converting_at_time(t)
        print(t, uc)
        assert uc.n >= 0.0 and uc.n <= 1.0
        assert uc.s >= 0.0

    print("cumulative c(t) as a function of t")
    for t in [0, 1, 2, 3, 4, 5, 7, 10, 20, 50, 100]:
        uc = model.expected_probability_of_converting_within_time(t)
        print(t, uc)
        assert uc.n >= 0.0 and uc.n <= 1.0
        assert uc.s >= 0.0
