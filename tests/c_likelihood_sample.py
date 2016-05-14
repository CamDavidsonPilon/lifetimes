
import lifetimes.generate_data as gen
from lifetimes import models,estimation
from lifetimes.data_compression import compress_data,compress_session_purchase_data
import ctypes as ct
import timeit
import math



def test_BGBB_likelihood_compressed():
    T = 10
    size = 1000
    params = {'alpha': 0.56, 'beta': 1.17, 'gamma': 0.38, 'delta': 1.13}

    data = gen.bgbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], size=size)

    data = compress_data(data)

    model = models.BGBBModel()

    n = len(data)
    n_samples = ct.c_int(n)
    int_n_size_array = ct.c_float * n

    x = int_n_size_array(*data['frequency'])
    tx = int_n_size_array(*data['recency'])
    T = int_n_size_array(*data['T'])
    #N = int_n_size_array(*data['N'])

    start_c = timeit.default_timer()


    likelihood_c = model.fitter._c_negative_log_likelihood(params.values(),x,tx,T,None,n_samples)
    c_time = timeit.default_timer() - start_c
    print "C_time: " + str(c_time)
    print "Likelihood: " + str(likelihood_c)

    start_py = timeit.default_timer()
    likelihood_py = model.fitter._negative_log_likelihood(params.values(),data['frequency'],data['recency'],data['T'],penalizer_coef=0, N = None)
    py_time = timeit.default_timer() - start_py
    print "Py_time: " + str(py_time)
    print "Likelihood: " + str(likelihood_py)
    assert py_time > 3 * c_time
    assert math.fabs(likelihood_c - likelihood_py) < 10**-6

def test_BGBBBB_likelihood_compressed():
    T = 10
    size = 1000
    params = {'alpha': 0.56, 'beta': 1.17, 'gamma': 0.38, 'delta': 1.13, 'epsilon': 0.1, 'zeta': 2.5}

    data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], params['epsilon'], params['zeta'], size=size)

    data = compress_session_purchase_data(data)

    model = models.BGBBBBModel()

    n = len(data)
    n_samples = ct.c_int(n)
    int_n_size_array = ct.c_float * n

    x = int_n_size_array(*data['frequency'])
    tx = int_n_size_array(*data['recency'])
    T = int_n_size_array(*data['T'])
    N = int_n_size_array(*data['N'])
    xp = int_n_size_array(*data['frequency_purchases'])

    start_c = timeit.default_timer()
    likelihood_c = model.fitter._c_negative_log_likelihood(params.values(),x,tx,T,xp,N,n_samples,estimation.c_lib)
    c_time = timeit.default_timer() - start_c
    print "C_time: " + str(c_time)
    print "Likelihood: " + str(likelihood_c)

    start_py = timeit.default_timer()
    likelihood_py = model.fitter._negative_log_likelihood(params.values(),data['frequency'],data['recency'],data['T'], data['frequency_purchases'], penalizer_coef=0, N = data['N'])
    py_time = timeit.default_timer() - start_py
    print "Py_time: " + str(py_time)
    print "Likelihood: " + str(likelihood_py)
    assert py_time > 3 * c_time
    assert math.fabs(likelihood_c - likelihood_py) < 10**-6

#def test_BGBBBB_fit(title,c,data):


def fit_no_N(model,title,c,data):

    start_c = timeit.default_timer()
    model.fit(data['frequency'], recency = data['recency'], T = data['T'], N = None, frequency_purchases=data['frequency_purchases'], c_fit=c)
    print title + str(timeit.default_timer() - start_c)
    print model.params

def fit(model,title,c,data):

    start_c = timeit.default_timer()
    model.fit(data['frequency'], recency=data['recency'], T=data['T'], N=data['N'], c_fit=c)
    print title + str(timeit.default_timer() - start_c)
    print model.params







