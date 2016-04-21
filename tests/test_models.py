import math

import pytest

import lifetimes.generate_data as gen
from lifetimes import models
from lifetimes.models import extract_frequencies


@pytest.mark.models
def test_model_fitting_and_simulation():
    T = 40
    r = 0.24
    alpha = 4.41
    a = 0.79
    b = 2.43

    data = gen.beta_geometric_nbd_model(T, r, alpha, a, b, size=1000)

    model = models.BetaGeoModel()
    data = model.generateData(T,{'r':r,'alpha':alpha,'a':a,'b':b},1000)
    model.fit(data['frequency'], data['recency'], data['T'])

    print "After fitting"
    print model.params
    print model.params_C
    print model.sampled_parameters

    numerical_metrics = model.evaluate_metrics_with_simulation(N = 100,t = 100)

    print "After simulating"
    numerical_metrics.dump()


@pytest.mark.models
def test_model_fitting_simulation_comparison_with_analytical_numbers():
    T = 40
    r = 0.24
    alpha = 4.41
    a = 0.79
    b = 2.43

    t = 100
    N = 1000

    model = models.ParetoNBDModel()
    par_gen = {'r': r, 'alpha': alpha, 's': a, 'beta': b}
    data = model.generateData(t,par_gen,size = 1000)
    model.fit(data['frequency'], data['recency'], data['T'])

    print "After fitting"
    print model.fitter
    print model.param_names
    print model.params
    print model.params_C

    Xt = model.evaluate_metrics_with_simulation(N, t, N_sim=10, max_x=10)

    print "After simulating"
    Xt.dump()

    print "Reference probabilities"
    ref_p = []
    #for x in range(Xt.length()):
        #ref_p.append(model.fitter.probability_of_n_purchases_up_to_time(t, x))
    print ref_p

    # compare with analytical formulas

    print "Compare expected values E[x]:"
    print "expected : " + str(model.fitter.expected_number_of_purchases_up_to_time(t))
    Ex, Ex_err = Xt.expected_x()
    print "MC value : " + str(Ex) + " +/- " + str(Ex_err)

    # divide data in calibration/test and compare results
    test_data = model.generateData(t,par_gen,size = N)
    test_fx = extract_frequencies(test_data)

    print "Empirical frequencies: " + str(test_fx)


@pytest.mark.models
def test_model_fitting_compare_simple_frequencies():
    T = 10
    r = 0.24
    alpha = 4.41
    a = 0.79
    b = 2.43

    data1 = gen.beta_geometric_nbd_model(T, r, alpha, a, b, size=10000)
    data2 = gen.beta_geometric_nbd_model(T, r, alpha, a, b, size=10000)

    fx1 = extract_frequencies(data1)
    fx2 = extract_frequencies(data2)

    print fx1
    print fx2


@pytest.mark.models
def test_NumericalMetrics():
    p_x = [0.1, 0.2, 0.7]
    p_x_err = [0.1, 0.1, 0.1]
    nm = models.NumericalMetrics(p_x, p_x_err)

    nm.dump()

    Ex, Ex_err = nm.expected_x()
    Ex_exp, Ex_err_exp = 0.2 + 1.4, math.sqrt((1 * 0.1) ** 2 + (2 * 0.1) ** 2)

    assert Ex > 1
    assert Ex < 2
    assert Ex_err > 0
    assert Ex == Ex_exp
    assert Ex_err == Ex_err_exp
