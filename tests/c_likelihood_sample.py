from __future__ import print_function

import lifetimes.generate_data as gen
from lifetimes import models,estimation
from lifetimes.data_compression import compress_data,compress_session_purchase_data
import ctypes as ct
import timeit
import math
import gc



def test_BGBB_likelihood_compressed(T,size):
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
    N = int_n_size_array(*data['N'])

    start_c = timeit.default_timer()

    likelihood_c = model.fitter._c_negative_log_likelihood(params.values(),x,tx,T,N,n_samples)
    c_time = timeit.default_timer() - start_c
    print("C_time: " + str(c_time))
    print("Likelihood: " + str(likelihood_c))

    start_py = timeit.default_timer()
    likelihood_py = model.fitter._negative_log_likelihood(params.values(),data['frequency'],data['recency'],data['T'],penalizer_coef=0, N = N)
    py_time = timeit.default_timer() - start_py
    print("Py_time: " + str(py_time))
    print("Likelihood: " + str(likelihood_py))
    #assert py_time > 3 * c_time
    #assert math.fabs(likelihood_c - likelihood_py) < 10**-6

def test_BGBBBB_likelihood_compressed_optimized(T,size):

    params = {'alpha': 0.56, 'beta': 1.17, 'gamma': 0.38, 'delta': 1.13, 'epsilon': 0.1, 'zeta': 2.5}

    data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], params['epsilon'], params['zeta'], size=size)

    data = compress_session_purchase_data(data)

    model = models.BGBBBBModel()

    start_py = timeit.default_timer()
    likelihood_py = model.fitter._negative_log_likelihood(params.values(),data['frequency'],data['recency'],data['T'], data['frequency_purchases'], penalizer_coef=0, N = data['N'])
    py_time = timeit.default_timer() - start_py
    print("Py_time: " + str(py_time))
    print("Likelihood: " + str(likelihood_py))

    gc.collect()

    c_data = []
    for x, tx, T, N, xp in zip(data['frequency'], data['recency'], data['T'], data['frequency_purchases'],data['N']):
        c_data.append(x)
        c_data.append(tx)
        c_data.append(T)
        c_data.append(N)
        c_data.append(xp)

    n = len(c_data)
    n_samples = ct.c_int(len(data))
    int_n_size_array = ct.c_float * n

    c_data = int_n_size_array(*c_data)

    start_c = timeit.default_timer()
    likelihood_c = model.fitter._c_likelihood_optimized(params.values(),c_data,n_samples)
    c_time = timeit.default_timer() - start_c
    print("C_opt_time: " + str(c_time))
    print("Likelihood: " + str(likelihood_c))

    gc.collect()

    n = len(data)
    n_samples = ct.c_int(n)
    int_n_size_array = ct.c_float * n

    x = int_n_size_array(*data['frequency'])
    tx = int_n_size_array(*data['recency'])
    T = int_n_size_array(*data['T'])
    N = int_n_size_array(*data['N'])
    xp = int_n_size_array(*data['frequency_purchases'])

    start_c = timeit.default_timer()
    likelihood_c = model.fitter._c_negative_log_likelihood(params.values(),x,tx,T,xp,N,n_samples)
    c_time = timeit.default_timer() - start_c
    print("C_time: " + str(c_time))
    print("Likelihood: " + str(likelihood_c))

    gc.collect()

    start_c = timeit.default_timer()
    likelihood_c = model.fitter._c_negative_log_likelihood_float(params.values(), x, tx, T, xp, N, n_samples)
    c_time = timeit.default_timer() - start_c
    print("C_float_time: " + str(c_time))
    print("Likelihood: " + str(likelihood_c))


    #assert py_time > 3 * c_time
    #assert math.fabs(likelihood_c - likelihood_py) < 10**-6

def test_BGBBBB_likelihood_compressed(T,size):

    params = {'alpha': 0.56, 'beta': 1.17, 'gamma': 0.38, 'delta': 1.13, 'epsilon': 0.1, 'zeta': 2.5}

    data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], params['epsilon'], params['zeta'], size=size)

    data = compress_session_purchase_data(data)

    model = models.BGBBBBModel()

    start_c = timeit.default_timer()
    n = len(data)
    n_samples = ct.c_int(n)
    int_n_size_array = ct.c_float * n


    x = int_n_size_array(*data['frequency'])
    tx = int_n_size_array(*data['recency'])
    T = int_n_size_array(*data['T'])
    N = int_n_size_array(*data['N'])
    xp = int_n_size_array(*data['frequency_purchases'])
    print("Data_time: " + str(timeit.default_timer() - start_c))

    start_exec = timeit.default_timer()
    likelihood_c = model.fitter._c_negative_log_likelihood(params.values(),x,tx,T,xp,N,n_samples)
    end_c = timeit.default_timer()
    print("Exec: " + str(end_c - start_exec))
    print("C_time: " + str(end_c - start_c))
    print("Likelihood: " + str(likelihood_c))

    start_py = timeit.default_timer()
    likelihood_py = model.fitter._negative_log_likelihood(params.values(),data['frequency'],data['recency'],data['T'], data['frequency_purchases'], penalizer_coef=0, N = data['N'])
    py_time = timeit.default_timer() - start_py
    print("Py_time: " + str(py_time))
    print("Likelihood: " + str(likelihood_py))
    #assert py_time > 3 * c_time
    #assert math.fabs(likelihood_c - likelihood_py) < 10**-6

def test_BGBBBB_fit():
    def test_fit(model, c, data):
        start = timeit.default_timer()
        model.fit(data['frequency'], recency=data['recency'], T=data['T'], N=data['N'], c_fit=c)
        time = timeit.default_timer() - start
        return time, model.params

    T = 10
    size = 1000
    params = {'alpha': 0.56, 'beta': 1.17, 'gamma': 0.38, 'delta': 1.13, 'epsilon': 0.1, 'zeta': 2.5}
    data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], params['epsilon'],
                            params['zeta'], size=size)

    c_time,c_params = test_fit(models.BGBBBBModel(),True,data)
    py_time,py_params = test_fit(models.BGBBBBModel(),False,data)

    assert c_time*3 < py_time
    for param_c,param_py in zip(c_params.values(),py_params.values()):
        assert math.fabs(param_c - param_py) < 10**-3


def test_BGBBBB_compressed_fit():
    def test_fit_compressed(model,c,data):
        start = timeit.default_timer()
        model.fit(data['frequency'], recency = data['recency'], T = data['T'], N = data['N'], frequency_purchases=data['frequency_purchases'], c_fit=c,bootstrap_size=3,iterative_fitting=0)
        time = timeit.default_timer() - start
        print(model.params_C)
        return time, model.fitter.params_

    T = 30
    size = 10000
    params = {'alpha': 0.56, 'beta': 1.17, 'gamma': 0.38, 'delta': 1.13, 'epsilon': 0.1, 'zeta': 2.5}
    data = gen.bgbbbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], params['epsilon'],
                            params['zeta'], size=size)

    compressed_data = compress_session_purchase_data(data)
    c_time,c_params = test_fit_compressed(models.BGBBBBModel(), True,compressed_data)
    py_time,py_params = test_fit_compressed(models.BGBBBBModel(), False, compressed_data)

    print(c_time)
    print(py_time)
    print(py_params)
    #for param_c, param_py in zip(c_params.values(), py_params.values()):
     #   assert math.fabs(param_c - param_py) < 10**-3


test_BGBBBB_compressed_fit()


