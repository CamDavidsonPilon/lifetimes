import pytest
import math
import lifetimes.generate_data as gen
import numpy as np
import lifetimes.estimation as est
from lifetimes import models
from lifetimes.data_compression import compress_bgext_data
from lifetimes.utils import is_almost_equal, is_same_order
from uncertainties import correlation_matrix, ufloat


@pytest.mark.BGExt
def test_BGExt_generation():
    params = {'alpha': 0.3, 'beta': 1.5}
    probs = (1,)

    gen_data = gen.bgext_model(5, params['alpha'], params['beta'], probs=probs, size=10)

    assert len(gen_data) == 10
    assert 'T' in gen_data
    assert 'frequency' in gen_data
    assert 'theta' in gen_data
    assert 'alt_state' in gen_data
    print gen_data

    print compress_bgext_data(gen_data)

    gen_data = gen.bgext_model([5, 5, 1, 1], params['alpha'], params['beta'], probs=probs, size=10)

    assert len(gen_data) == 4
    assert 'T' in gen_data
    assert 'frequency' in gen_data
    assert 'theta' in gen_data
    assert 'alt_state' in gen_data
    print gen_data


@pytest.mark.BGExt
def test_likelyhood():
    params = {'alpha': 1.2, 'beta': 0.7}

    # test simgle number as inputs
    freq = 1
    T = 2
    ll = est.BGFitter._negative_log_likelihood(params.values(), freq, T, penalizer_coef=0, N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test simgle number as inputs
    freq = 1
    T = 1
    ll = est.BGFitter._negative_log_likelihood(params.values(), freq, T, penalizer_coef=0, N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test arrays as inputs
    freq = [1, 0]
    T = [2, 2]
    ll = est.BGFitter._negative_log_likelihood(params.values(), freq, T, penalizer_coef=0, N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test np.arrays as inputs
    freq = np.array([1, 0])
    T = np.array([2, 2])
    ll1 = est.BGFitter._negative_log_likelihood(params.values(), freq, T, penalizer_coef=0, N=None)

    assert ll == ll1
    assert isinstance(ll1, float) or isinstance(ll1, int)

    # test np.arrays as inputs
    freq = np.array([1, 0])
    T = np.array([1, 1])
    ll1 = est.BGFitter._negative_log_likelihood(params.values(), freq, T, penalizer_coef=0, N=None)

    assert isinstance(ll1, float) or isinstance(ll1, int)


@pytest.mark.BGExt
def test_BG_fitting_compressed_or_not():
    T = 10
    size = 1000
    params = {'alpha': 0.3, 'beta': 3.7}

    data = gen.bgext_model(T, params['alpha'], params['beta'], size=size)

    print data

    compressed_data = compress_bgext_data(data)

    fitter = est.BGFitter(penalizer_coef=0.1)
    fitter_compressed = est.BGFitter(penalizer_coef=0.1)

    fitter.fit(data['frequency'], data['T'], initial_params=params.values())
    fitter_compressed.fit(compressed_data['frequency'], compressed_data['T'],
                          N=compressed_data['N'], initial_params=params.values())

    print params
    print fitter.params_
    print fitter_compressed.params_

    for par_name in params.keys():
        assert math.fabs(fitter.params_[par_name] - fitter_compressed.params_[par_name]) < 0.00001


@pytest.mark.BGExt
def test_BG_additional_functions():
    T = 10
    size = 1000
    params = {'alpha': 0.3, 'beta': 3.7}

    data = gen.bgext_model(T, params['alpha'], params['beta'], size=size)

    print data

    data = compress_bgext_data(data)

    fitter = est.BGFitter(penalizer_coef=0.1)

    fitter.fit(data['frequency'], data['T'], N=data['N'], initial_params=params.values())

    print "Generation params"
    print params

    print "Fitted params"
    print fitter.params_

    print "E[X(t)] as a function of t"
    for t in [1, 10, 100, 1000, 10000]:
        Ex = fitter.expected_number_of_purchases_up_to_time(t)
        covariance_matrix = np.cov(np.vstack([[(params['alpha'] * 0.1) ** 2, 0], [0, (params['beta'] * 0.1) ** 2]]))
        Ex_err = fitter.expected_number_of_purchases_up_to_time_error(t, covariance_matrix)
        print t, Ex, Ex / t, Ex_err
        assert Ex >= 0
        assert Ex_err >= 0

    t = 10
    print "P[X(t) = n] as a function of n, t = " + str(t)
    tot_prob = 0.0
    for n in range(t + 1):
        prob = fitter.probability_of_n_purchases_up_to_time(t, n)
        print n, prob
        tot_prob += prob
        assert 1 >= prob >= 0

    assert math.fabs(tot_prob - 1.0) < 0.00001


@pytest.mark.BGExt
def test_BG_integration_in_models():
    T = 10
    size = 1000
    params = {'alpha': 0.17, 'beta': 1.18}

    data = gen.bgext_model([1] * 300 + [2] * 200 + [3] * 180 + [4] * 37, params['alpha'],
                           params['beta'])  # , size=size)

    print data

    data = compress_bgext_data(data)

    model = models.BGModel(penalizer_coef=0.1)

    model.fit(data['frequency'], data['T'], bootstrap_size=10, N=data['N'],
              initial_params=params.values())

    assert model.is_ready()
    assert model.good_fit()

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


@pytest.mark.BGExt
def test_BG_on_simil_real_data():
    T = 10
    size = 1000
    params = {'alpha': 0.17, 'beta': 1.18}

    # let's take a case similar to Spy Calc Free:
    # data = gen.bgext_model([1] * 300 + [2] * 200 + [3] * 180 + [4] * 37, params['alpha'], params['beta']) #, size=size)
    data = gen.bgext_model([1] * 3000 + [2] * 2000 + [3] * 1800 + [4] * 370, params['alpha'],
                           params['beta'])  # , size=size)

    # data = pd.read_csv("/Users/marcomeneghelli/Desktop/SCF_data.csv")
    data = compress_bgext_data(data)

    model = models.BGModel(penalizer_coef=0.1)

    model.fit(data['frequency'], data['T'], bootstrap_size=10, N=data['N'],
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


@pytest.mark.BGExt
def test_BG_integration_in_models_with_uncertainties():
    T = 10
    size = 1000
    params = {'alpha': 0.17, 'beta': 1.18}

    data = gen.bgext_model([1] * 300 + [2] * 200 + [3] * 180 + [4] * 37, params['alpha'],
                           params['beta'])  # , size=size)
    data = gen.bgext_model(T, params['alpha'], params['beta'], size=size)

    data = compress_bgext_data(data)

    model = models.BGModel(penalizer_coef=0.1)

    model.fit(data['frequency'], data['T'], bootstrap_size=10, N=data['N'],
              initial_params=params.values())

    print "Generation params"
    print params

    print "Fitted params"
    print model.params
    print model.params_C

    print "Uncertain parameters"
    print model.uparams

    print "E[X(t)] as a function of t"
    for t in [0, 1, 2, 3, 4, 5, 7, 10, 20, 50, 100, 1000, 10000]:
        Ex, Ex_err = model.expected_number_of_purchases_up_to_time_with_errors(t)
        print t, Ex, Ex_err
        assert Ex >= 0
        assert Ex_err >= 0

        uEx = model.expected_number_of_purchases_up_to_time(t)
        print t, uEx
        assert is_almost_equal(Ex, uEx.n)
        assert is_same_order(Ex_err, uEx.s)

    t = 10
    print "E[X(t) = n] as a function of n, t = " + str(t)
    tot_prob = 0.0
    for n in range(t + 1):
        prob = model.fitter.probability_of_n_purchases_up_to_time(t, n)
        print n, prob
        tot_prob += prob
        assert 1 >= prob >= 0

        uprob = model.probability_of_n_purchases_up_to_time(t, n)
        print uprob
        assert is_almost_equal(uprob.n, prob)

    assert math.fabs(tot_prob - 1.0) < 0.00001


@pytest.mark.BGExt
def test_correlations_of_uparams_and_derivatives():
    T = 10
    size = 100
    params = {'alpha': 0.17, 'beta': 1.18}

    data = gen.bgext_model(T, params['alpha'], params['beta'], size=size)
    data = compress_bgext_data(data)

    model = models.BGModel(penalizer_coef=0.1)
    model.fit(data['frequency'], data['T'], bootstrap_size=10, N=data['N'],
              initial_params=params.values())

    print "Generation params"
    print params

    print "Fitted params"
    print model.params
    print model.params_C

    print "Uncertain parameters"
    print model.uparams

    assert is_almost_equal(correlation_matrix([model.uparams['alpha'], model.uparams['alpha']])[0, 1], 1.0)
    assert 1.0 > correlation_matrix([model.uparams['alpha'] + ufloat(1, 1), model.uparams['alpha']])[0, 1] > 0.0

    # stub of profile
    p1 = model.expected_number_of_purchases_up_to_time(1)
    p2 = model.expected_number_of_purchases_up_to_time(2)

    assert 1.0 > correlation_matrix([p1, p2])[0, 1] > 0.0

    # stub of profile
    p1 = model.expected_number_of_purchases_up_to_time(1)
    p2 = model.expected_number_of_purchases_up_to_time(10)

    assert 1.0 > correlation_matrix([p1, p2])[0, 1] > 0.0

    # stub of profile
    p1 = model.expected_number_of_purchases_up_to_time(1)
    p2 = model.expected_number_of_purchases_up_to_time(100)

    assert 1.0 > correlation_matrix([p1, p2])[0, 1] > 0.0

    # print [correlation_matrix([p1, model.expected_number_of_purchases_up_to_time(t)])[0, 1] for t in range(1,100)]
