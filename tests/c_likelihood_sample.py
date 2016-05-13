
import lifetimes.generate_data as gen
from lifetimes import models
from lifetimes.data_compression import compress_data
import ctypes as ct
import timeit

def test_BGBB_integration_in_models(c_likelihood_lib, title):
    T = 10
    size = 10000
    params = {'alpha': 0.56, 'beta': 1.17, 'gamma': 0.38, 'delta': 1.13}

    data = gen.bgbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], size=size)

    data = compress_data(data)

    model = models.BGBBModel()
    start_c = timeit.default_timer()
    model.fit(data['frequency'], data['recency'], data['T'], bootstrap_size=20, N=data['N'],
              initial_params=params.values(),c_likelihood_lib = c_likelihood_lib)
    print title + str(timeit.default_timer() - start_c)
    print "Generation params"
    print params

    print "Fitted params"
    print model.params
    print model.params_C

if __name__ == '__main__':
    #locate the library
    c_lib = ct.CDLL('./c_utilities/betalib.so')
    #set return type of your function
    c_lib.bgbb_likelihood.restype = ct.c_double
    
    test_BGBB_integration_in_models(c_lib,"C_Time: ")

    test_BGBB_integration_in_models(None,"Py_Time: ")




