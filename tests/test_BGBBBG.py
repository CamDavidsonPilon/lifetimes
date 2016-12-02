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

    print params
    # print fitter.params_
    print fitter_compressed.params_
    tot = 0
    fitted_conv = []
    for t in range(31):
        res = fitter_compressed.expected_probability_of_converting_at_time(t)
        tot += res
        fitted_conv.append(res)
    print tot

    conversion_instants = data['time_first_purchase']
    real_conv = [0] * 31
    for t in conversion_instants:
        if not math.isnan(t):
            real_conv[int(t)] += 1.0
    real_conv = [x / len(conversion_instants) for x in real_conv]

    for r, f in zip(real_conv, fitted_conv):
        print r, " - ", f

    print sum(real_conv), " - ", sum(fitted_conv)


    # for par_name in params.keys():
    # assert math.fabs(fitter.params_[par_name] - fitter_compressed.params_[par_name]) < 0.00001


@pytest.mark.BGBBBB
def test_BGBBBG_fitting_compressed_or_not():
    T = 10
    size = 100

    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    data = gen.bgbbbg_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                            params['epsilon'],
                            params['zeta'], size=size, time_first_purchase=True)

    compressed_data = compress_session_session_before_conversion_data(data)

    # fitter = est.BGBBBGFitter()
    model = mod.BGBBBGModel()
    # fitter.fit(data['frequency'], data['recency'], data['T'], data['frequency_before_conversion'],
    #           initial_params=params.values())
    model.fit(frequency=compressed_data['frequency'], recency=compressed_data['recency'], T=compressed_data['T'],
              frequency_before_conversion=compressed_data['frequency_before_conversion'],
              N=compressed_data['N'], initial_params=params.values())

    print params
    # print fitter.params_
    print model.params
    tot = 0
    fitted_conv = []

    for t in range(30):
        print model.expected_probability_of_converting_at_time_with_error(t)


###test_BGBBBGExt_fitting_compressed_or_not()