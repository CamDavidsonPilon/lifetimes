import pytest
import math
import lifetimes.generate_data as gen
import numpy as np
import lifetimes.estimation as est
from lifetimes.data_compression import compress_session_purchase_data
from lifetimes.data_compression import filter_data_by_T
import timeit
from lifetimes import models
import pandas as pd


@pytest.mark.BGBBBB
def test_BGBBBB_generation():
    T = 100
    size = 100

    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    gen_data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                                params['epsilon'],
                                params['zeta'], size=size)

    assert len(gen_data) == 100
    assert 'T' in gen_data
    assert 'frequency' in gen_data
    assert 'frequency_purchases' in gen_data
    assert 'recency' in gen_data
    assert 'p' in gen_data
    assert 'pi' in gen_data
    assert 'theta' in gen_data
    assert 'alive' in gen_data


@pytest.mark.BGBBBB
def test_BGBBBB_generation_with_time_first_purchase():
    T = 100
    size = 10000

    params = {'alpha': 0.48, 'beta': 0.96, 'gamma': 0.25, 'delta': 1.3, 'epsilon': 1.0, 'zeta': 10.0}

    gen_data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
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
    # conversion_data_frame.to_csv("/Users/marcomeneghelli/Desktop/conversion_simulation.csv")


@pytest.mark.BGBBBB
def test_BGBBBB_generation_with_time_death():
    T = 600
    size = 100000

    params = {'alpha': 0.48, 'beta': 0.96, 'gamma': 0.25, 'delta': 1.3, 'epsilon': 1.0, 'zeta': 10.0}

    gen_data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                                params['epsilon'],
                                params['zeta'], size=size, death_time=True)

    assert 'death_time' in gen_data

    # build conversion profile
    ts = range(T + 1)
    cs = [0] * len(ts)
    cs_err = [0] * len(ts)
    dt_column = gen_data['death_time']
    N = len(dt_column)
    for t in ts:
        n = float(len(dt_column[dt_column == t]))
        p = n / N
        cs[t] = p
        cs_err[t] = np.sqrt(1.0 / N * p * (1.0 - p))



@pytest.mark.BGBBBB
def test_likelyhood():
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    # test simgle number as inputs
    freq = 1
    freq_purch = 1
    rec = 1
    T = 2
    ll = est.BGBBBBFitter._negative_log_likelihood(params.values(), freq, rec, T, freq_purch, penalizer_coef=0, N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test arrays as inputs
    freq = [1, 0]
    freq_purch = [1, 1]
    rec = [1, 0]
    T = [2, 2]
    ll = est.BGBBBBFitter._negative_log_likelihood(params.values(), freq, rec, T, freq_purch, penalizer_coef=0, N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test np.arrays as inputs
    freq = np.array([1, 0])
    freq_purch = np.array([1, 1])
    rec = np.array([1, 0])
    T = np.array([2, 2])
    ll1 = est.BGBBBBFitter._negative_log_likelihood(params.values(), freq, rec, T, freq_purch, penalizer_coef=0, N=None)

    assert ll == ll1
    assert isinstance(ll1, float) or isinstance(ll1, int)


@pytest.mark.BGBBBB
def test_BGBB_fitting_compressed_or_not():
    T = 30
    size = 1000

    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                            params['epsilon'],
                            params['zeta'], size=size)

    compressed_data = compress_session_purchase_data(data)

    fitter = est.BGBBBBFitter()
    fitter_compressed = est.BGBBBBFitter()

    fitter.fit(data['frequency'], data['recency'], data['T'], data['frequency_purchases'],
               initial_params=params.values())
    fitter_compressed.fit(compressed_data['frequency'], compressed_data['recency'], compressed_data['T'],
                          compressed_data['frequency_purchases'],
                          N=compressed_data['N'], initial_params=params.values())

    print params
    print fitter.params_
    print fitter_compressed.params_

    for par_name in params.keys():
        assert math.fabs(fitter.params_[par_name] - fitter_compressed.params_[par_name]) < 0.00001


@pytest.mark.BGBBBB
def test_BGBBBB_additional_functions():
    T = 10
    size = 100
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                            params['epsilon'],
                            params['zeta'], size=size)

    compressed_data = compress_session_purchase_data(data)

    fitter = est.BGBBBBFitter()

    fitter.fit(compressed_data['frequency'], compressed_data['recency'], compressed_data['T'],
               compressed_data['frequency_purchases'],
               N=compressed_data['N'], initial_params=params.values())

    print "Generation params"
    print params

    print "Fitted params"
    print fitter.params_

    print "E[X(t)] as a function of t"
    for t in [0, 1, 10, 100, 1000, 10000]:
        Ex = fitter.expected_number_of_sessions_up_to_time(t)
        covariance_matrix = np.cov(np.vstack([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        Ex_err = fitter.expected_number_of_sessions_up_to_time_error(t, covariance_matrix)
        print t, Ex, Ex_err
        assert Ex >= 0
        assert Ex_err >= 0

    print "E[X_p(t)] as a function of t"
    for t in [0, 1, 10, 100, 1000, 10000]:
        Ex = fitter.expected_number_of_purchases_up_to_time(t)
        covariance_matrix = np.cov(np.vstack(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]]))
        Ex_err = fitter.expected_number_of_purchases_up_to_time_error(t, covariance_matrix)
        print t, Ex, Ex_err
        assert Ex >= 0
        assert Ex_err >= 0


@pytest.mark.BGBBBB
def test_BGBBNN_fitting_with_different_T_windows():
    size = 50
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    est_params = {}

    T = range(1, 100 + 1)
    data = gen.bgbbbb_model(T[0], params['alpha'], params['beta'], params['gamma'], params['delta'], params['epsilon'],
                            params['zeta'], size=size)

    for i in range(2, len(T)):
        data_addendum = gen.bgbbbb_model(T[i], params['alpha'], params['beta'], params['gamma'], params['delta'],
                                         params['epsilon'], params['zeta'],
                                         size=size)
        data = data.append(data_addendum)

    data.index = range(len(data.index))
    data = compress_session_purchase_data(data)

    fitter = est.BGBBBBFitter()

    T1s = [1, 15, 30, 45, 60, 75]
    deltas = [30, 50, 60]
    for T1 in T1s:
        est_params[T1] = {}
        for delta in deltas:
            T2 = T1 + delta
            filtered_data = filter_data_by_T(data, T1, T2)
            fitter.fit(filtered_data['frequency'], filtered_data['recency'], filtered_data['T'],
                       filtered_data['frequency_purchases'], N=filtered_data['N'])
            est_params[T1][delta] = fitter.params_

    print est_params

    for T1 in T1s:
        for delta in deltas:
            current_params = est_params[T1][delta]
            assert 'alpha' in current_params
            assert 'beta' in current_params
            assert 'gamma' in current_params
            assert 'delta' in current_params
            assert 'epsilon' in current_params
            assert 'zeta' in current_params

            assert math.fabs(current_params['alpha'] - params['alpha']) < 6 * params['alpha']


@pytest.mark.BGBBBB
def test_BGBBBB_fitting_time():
    T = 100
    sizes = [10, 100, 1000]
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    compressed_data = {}
    for size in sizes:
        data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], params['epsilon'],
                                params['zeta'], size=size)
        compressed_data[size] = compress_session_purchase_data(data)

    times = {}
    lengths = {}
    for size in sizes:
        fitter = est.BGBBBBFitter()
        start_time = timeit.default_timer()
        fitter.fit(compressed_data[size]['frequency'], compressed_data[size]['recency'], compressed_data[size]['T'],
                   compressed_data[size]['frequency_purchases'],
                   N=compressed_data[size]['N'], initial_params=params.values())
        t1 = timeit.default_timer() - start_time
        times[size] = t1
        lengths[size] = len(compressed_data[size])

    print params
    print fitter.params_
    print times
    print lengths


@pytest.mark.BGBBBB
def test_BGBBBB_integration_in_models():
    T = 30
    size = 1000
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], params['epsilon'],
                            params['zeta'], size=size)

    data = compress_session_purchase_data(data)

    model = models.BGBBBBModel()

    model.fit(data['frequency'], data['recency'], data['T'], frequency_purchases=data['frequency_purchases'],
              bootstrap_size=10, N=data['N'],
              initial_params=params.values())

    print "Generation params"
    print params

    print "Fitted params"
    print model.params
    print model.params_C

    print "E[X(t)] as a function of t"
    for t in [0, 1, 10, 100, 1000, 10000]:
        Ex, Ex_err = model.expected_number_of_sessions_up_to_time_with_errors(t)
        print t, Ex, Ex_err
        # assert Ex >= 0
        # assert Ex_err >= 0

    print "E[X_p(t)] as a function of t"
    for t in [0, 1, 10, 100, 1000, 10000]:
        Ex, Ex_err = model.expected_number_of_purchases_up_to_time_with_errors(t)
        print t, Ex, Ex_err
        # assert Ex >= 0
        # assert Ex_err >= 0


@pytest.mark.BGBBBB
def test_BGBBBB_evaluate_metrics_with_simulation():
    T = 30
    size = 100
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], params['epsilon'],
                            params['zeta'], size=size)

    data = compress_session_purchase_data(data)

    model = models.BGBBBBModel()

    model.fit(data['frequency'], data['recency'], data['T'], frequency_purchases=data['frequency_purchases'],
              bootstrap_size=10, N=data['N'],
              initial_params=params.values())

    session_numerical_metrics = model.evaluate_metrics_with_simulation(1000, 100, 10, 50, tag='frequency')
    purchases_numerical_metrics = model.evaluate_metrics_with_simulation(1000, 100, 10, 50, tag='frequency_purchases')

    session_numerical_metrics.dump()
    purchases_numerical_metrics.dump()

    assert True
