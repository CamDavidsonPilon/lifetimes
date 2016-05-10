import pytest
import math
import lifetimes.generate_data as gen
import numpy as np
import lifetimes.estimation as est
from lifetimes.data_compression import compress_data
from lifetimes.data_compression import filter_data_by_T
import timeit
from scipy import special
from lifetimes import models


@pytest.mark.BGBB
def test_BGBB_generation():
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7}

    gen_data = gen.bgbb_model(5, params['alpha'], params['beta'], params['gamma'], params['delta'], size=10)

    assert len(gen_data) == 10
    assert 'T' in gen_data
    assert 'frequency' in gen_data
    assert 'recency' in gen_data
    assert 'p' in gen_data
    assert 'theta' in gen_data
    assert 'alive' in gen_data
    assert 'id' in gen_data


@pytest.mark.BGBB
def test_binomial_probs():
    p = 0.6

    n = 0
    N = 1000
    for i in range(N):
        success = np.random.random() <= p
        if success:
            n += 1
    assert float(n) / N > p - 5 * math.sqrt(1.0 / N * p * (1.0 - p))


@pytest.mark.BGBB
def test_likelyhood():
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7}

    # test simgle number as inputs
    freq = 1
    rec = 1
    T = 2
    ll = est.BGBBFitter._negative_log_likelihood(params.values(), freq, rec, T, penalizer_coef=0, N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test arrays as inputs
    freq = [1, 0]
    rec = [1, 0]
    T = [2, 2]
    ll = est.BGBBFitter._negative_log_likelihood(params.values(), freq, rec, T, penalizer_coef=0, N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test np.arrays as inputs
    freq = np.array([1, 0])
    rec = np.array([1, 0])
    T = np.array([2, 2])
    ll1 = est.BGBBFitter._negative_log_likelihood(params.values(), freq, rec, T, penalizer_coef=0, N=None)

    assert ll == ll1
    assert isinstance(ll1, float) or isinstance(ll1, int)


@pytest.mark.BGBB
def test_BGBB_new_likelyhood():
    a, b, g, d = 1.2, 0.7, 0.6, 2.7

    x = np.array([0, 1, 2, 3, 4, 5])
    tx = np.array([0, 1, 1, 1, 3, 5])
    T = np.array([5, 5, 5, 5, 5, 5])

    numerator = special.beta(a + x, b + T - x) * special.beta(g, d + T)

    max_i = (T - tx - 1).astype(int)
    for j in range(len(max_i)):
        xj = x[j]
        txj = tx[j]
        i = np.arange(max_i[j] + 1)
        numerator[j] += np.sum(special.beta(a + xj, b + txj - xj + i) * special.beta(g + 1, d + txj + i))

    first_term = special.beta(a + x, b + T - x) * special.beta(g, d + T)

    second_term = np.array([np.sum(
        [special.beta(a + x[j], b + tx[j] - x[j] + i) * special.beta(g + 1, d + tx[j] + i) for i in
         range(int(T[j] - tx[j] - 1 + 1))])
                            for j in range(len(x))])

    for ii in range(len(numerator)):
        assert numerator[ii] == first_term[ii] + second_term[ii]


@pytest.mark.BGBB
def test_BGBB_fitting_compressed_or_not():
    T = 10
    size = 1000
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7}

    data = gen.bgbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], size=size)

    compressed_data = compress_data(data)

    fitter = est.BGBBFitter()
    fitter_compressed = est.BGBBFitter()

    fitter.fit(data['frequency'], data['recency'], data['T'], initial_params=params.values())
    fitter_compressed.fit(compressed_data['frequency'], compressed_data['recency'], compressed_data['T'],
                          N=compressed_data['N'], initial_params=params.values())

    print params
    print fitter.params_
    print fitter_compressed.params_

    for par_name in params.keys():
        assert math.fabs(fitter.params_[par_name] - fitter_compressed.params_[par_name]) < 0.00001


@pytest.mark.BGBB
def test_BGBB_additional_functions():
    T = 10
    size = 100
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7}

    data = gen.bgbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], size=size)

    data = compress_data(data)

    fitter = est.BGBBFitter()

    fitter.fit(data['frequency'], data['recency'], data['T'], N=data['N'], initial_params=params.values())

    print "Generation params"
    print params

    print "Fitted params"
    print fitter.params_

    print "E[X(t)] as a function of t"
    for t in [0, 1, 10, 100, 1000, 10000]:
        Ex = fitter.expected_number_of_purchases_up_to_time(t)
        covariance_matrix = np.cov(np.vstack([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        Ex_err = fitter.expected_number_of_purchases_up_to_time_error(t, covariance_matrix)
        print t, Ex, Ex_err
        assert Ex >= 0
        assert Ex_err >= 0

    t = 10
    print "E[X(t) = n] as a function of n, t = " + str(t)
    tot_prob = 0.0
    for n in range(t + 1):
        prob = fitter.probability_of_n_purchases_up_to_time(t, n)
        print n, prob
        tot_prob += prob
        assert 1 >= prob >= 0

    assert math.fabs(tot_prob - 1.0) < 0.00001


@pytest.mark.BGBB
def test_BGBB_fitting_with_different_T_windows():
    size = 10
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7}

    est_params = {}

    T = range(1, 100 + 1)
    data = gen.bgbb_model(T[0], params['alpha'], params['beta'], params['gamma'], params['delta'], size=size)

    for i in range(2, len(T)):
        data_addendum = gen.bgbb_model(T[i], params['alpha'], params['beta'], params['gamma'], params['delta'],
                                       size=size)
        data = data.append(data_addendum)

    data.index = range(len(data.index))
    data = compress_data(data)

    fitter = est.BGBBFitter()

    T1s = [1, 15, 30, 45, 60, 75]
    deltas = [30, 50, 60]
    for T1 in T1s:
        est_params[T1] = {}
        for delta in deltas:
            T2 = T1 + delta
            filtered_data = filter_data_by_T(data, T1, T2)
            fitter.fit(filtered_data['frequency'], filtered_data['recency'], filtered_data['T'], N=filtered_data['N'])
            est_params[T1][delta] = fitter.params_

    print est_params

    for T1 in T1s:
        for delta in deltas:
            current_params = est_params[T1][delta]
            assert 'alpha' in current_params
            assert 'beta' in current_params
            assert 'gamma' in current_params
            assert 'delta' in current_params

            assert math.fabs(current_params['alpha'] - params['alpha']) < 6 * params['alpha']


@pytest.mark.BGBB
def test_BGBB_fitting_time():
    T = 100
    sizes = [10, 100, 1000]
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7}

    compressed_data = {}
    for size in sizes:
        data = gen.bgbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], size=size)
        compressed_data[size] = compress_data(data)

    times = {}
    for size in sizes:
        fitter = est.BGBBFitter()
        start_time = timeit.default_timer()
        fitter.fit(compressed_data[size]['frequency'], compressed_data[size]['recency'], compressed_data[size]['T'],
                   N=compressed_data[size]['N'], initial_params=params.values())
        t1 = timeit.default_timer() - start_time
        times[size] = t1

    print times


@pytest.mark.BGBB
def test_BGBB_integration_in_models():
    T = 10
    size = 100
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7}

    data = gen.bgbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], size=size)

    data = compress_data(data)

    model = models.BGBBModel()

    model.fit(data['frequency'], data['recency'], data['T'], bootstrap_size=10, N=data['N'],
              initial_params=params.values())

    print "Generation params"
    print params

    print "Fitted params"
    print model.params
    print model.params_C

    print "E[X(t)] as a function of t"
    for t in [0, 1, 10, 100, 1000, 10000]:
        Ex, Ex_err = model.expected_number_of_purchases_up_to_time_with_errors(t)
        print t, Ex, Ex_err
        assert Ex >= 0
        assert Ex_err >= 0

    t = 10
    print "E[X(t) = n] as a function of n, t = " + str(t)
    tot_prob = 0.0
    for n in range(t + 1):
        prob = model.fitter.probability_of_n_purchases_up_to_time(t, n)
        print n, prob
        tot_prob += prob
        assert 1 >= prob >= 0

    assert math.fabs(tot_prob - 1.0) < 0.00001
