import math

import pytest

import lifetimes.generate_data as gen
from lifetimes import estimation
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

    model = models.BetaGeoModel(100, 10)
    model.fit(data['frequency'], data['recency'], data['T'])

    print "After fitting"
    print model.fitted_model
    print model.params
    print model.params_C
    print model.sampled_parameters

    model.simulate()

    print "After simulating"
    model.numerical_metrics.dump()


@pytest.mark.models
def test_model_fitting_simulation_comparison_with_analytical_numbers():
    T = 40
    r = 0.24
    alpha = 4.41
    a = 0.79
    b = 2.43

    t = 100
    N = 1000

    data = gen.beta_geometric_nbd_model(T, r, alpha, a, b, size=1000)

    model = models.BetaGeoModel(N, t)
    model.fit(data['frequency'], data['recency'], data['T'])

    ref_model = estimation.BetaGeoFitter()
    ref_model.fit(data['frequency'], data['recency'], data['T'])

    print "After fitting"
    print model.fitted_model
    print model.params
    print model.params_C
    print model.numerical_metrics

    model.simulate()

    print "After simulating"
    model.numerical_metrics.dump()

    print "Reference probabilities"
    ref_p = []
    for x in range(model.numerical_metrics.length()):
        ref_p.append(ref_model.probability_of_n_purchases_up_to_time(t, x))
    print ref_p

    # compare with analytical formulas

    print "Compare expected values E[x]:"
    print "expected : " + str(ref_model.expected_number_of_purchases_up_to_time(t))
    Ex, Ex_err = model.numerical_metrics.expected_x()
    print "MC value : " + str(Ex) + " +/- " + str(Ex_err)

    # divide data in calibration/test and compare results
    test_data = gen.beta_geometric_nbd_model(t, r, alpha, a, b, size=N)
    test_fx = extract_frequencies(test_data)

    print "Empirical frequencies: " + str(test_fx)


@pytest.mark.models
def test_model_fitting_compare_simple_frequencies():
    T = 10
    r = 0.24
    alpha = 4.41
    a = 0.79
    b = 2.43

    data1 = gen.beta_geometric_nbd_model(T, r, alpha, a, b, size=1000)
    data2 = gen.beta_geometric_nbd_model(T, r, alpha, a, b, size=1000)

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
