import math

import pytest
import numpy as np
import lifetimes.generate_data as gen
from lifetimes import models
from lifetimes.models import extract_frequencies


@pytest.mark.models
def test_models_fitting_and_simulation():
    t = 40
    params = [0.24,4.41,0.79,2.43]


    test_models = [models.ParetoNBDModel(),models.BetaGeoModel(),models.ModifiedBetaGeoModel()]
    for model in test_models:
        assert model.params is None
        assert model.params_C is None

    for model in test_models:
        fitted_model = _fit_and_simulate(model=model,parameters=params,t = t)
        assert fitted_model.params is not None
        assert fitted_model.params_C is not None


def _fit_and_simulate(model,parameters,t):
    data = model.generateData(t = t,size = 100, parameters = model.parameters_dictionary_from_list(parameters))
    model.fit(data['frequency'], data['recency'], data['T'])
    return model


@pytest.mark.models
def test_model_fitting_and_simulation():
    T = 40
    r = 0.24
    alpha = 4.41
    a = 0.79
    b = 2.43

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

    # compare with analytical formulas

    print "Compare expected values E[x]:"
    print "expected : " + str(model.fitter.expected_number_of_purchases_up_to_time(t))
    Ex, Ex_err = Xt.expected_x()
    print "MC value : " + str(Ex) + " +/- " + str(Ex_err)

    assert math.fabs(model.fitter.expected_number_of_purchases_up_to_time_error(t,model.params_C)  - EM_expected_number_of_purchases_up_to_time_error(model.fitter,t,model.params_C)) < 10**-6

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


def EM_expected_number_of_purchases_up_to_time_error(pareto, t, C):
    E = pareto.expected_number_of_purchases_up_to_time(t)
    r, alpha, s, beta = pareto._unload_params('r', 'alpha', 's', 'beta')

    d_r = E / r

    d_alpha = - E / alpha

    coeff = r / (alpha * (s - 1))
    exp_base = beta / (beta + t)

    d_s = beta * coeff * (- 1 / (s - 1) * (1 - exp_base ** (s - 1)) - exp_base ** (s - 1) * np.log(exp_base))
    d_beta = coeff * (1 - exp_base ** (s - 1) - beta * (s - 1) * exp_base ** (s - 2) * t / (beta + t) ** 2)

    j = np.matrix([d_r, d_alpha, d_s, d_beta])
    sigma = np.matrix(C)
    left = j * sigma
    square = left * j.transpose()
    return math.sqrt(float(square))