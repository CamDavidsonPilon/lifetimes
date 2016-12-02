import math

import pytest
import numpy as np
from lifetimes import models

t = 40
params = [0.24, 4.41, 0.79, 2.43]


@pytest.mark.models
def test_models_fitting_and_simulation():
    test_models = [models.ParetoNBDModel(), models.BetaGeoModel(), models.ModifiedBetaGeoModel()]
    for model in test_models:
        assert model.params is None
        assert model.params_C is None

    for model in test_models:
        fitted_model = _fit_and_simulate(model=model, parameters=params, t=t)
        assert fitted_model.params is not None
        assert fitted_model.params_C is not None


@pytest.mark.models
def test_Pareto_expected_number_of_purchases_with_error():
    fitted_model = _fit_and_simulate(models.ParetoNBDModel(), params, t)
    e_x, err_e_x = fitted_model.expected_number_of_sessions_up_to_time_with_errors(t)
    assert e_x is not None and err_e_x is not None
    assert math.fabs(err_e_x - EM_expected_number_of_purchases_up_to_time_error(fitted_model.fitter, t,
                                                                                fitted_model.params_C)) < 10 ** -6


@pytest.mark.models
def test_Pareto_expected_purchases_before_fitting():
    model = models.ParetoNBDModel()
    error_generated = False
    try:
        _, _ = model.expected_number_of_purchases_up_to_time_with_errors(1)
    except ValueError:
        error_generated = True
    assert error_generated


@pytest.mark.models
def test_params_covariance():
    model = _fit_and_simulate(models.ParetoNBDModel(), params, t)
    covariance_matrix = np.matrix(model.params_C)

    # test symmetry
    transpose_covariance = covariance_matrix.transpose()
    assert np.all(np.fabs(covariance_matrix - transpose_covariance) < 10 ** -6)

    # test_diagonal
    for i in range(len(covariance_matrix)):
        assert covariance_matrix.item((i, i)) >= 0


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


def EM_expected_number_of_purchases_up_to_time_error(pareto_fitter, t, C):
    E = pareto_fitter.expected_number_of_sessions_up_to_time(t)
    r, alpha, s, beta = pareto_fitter._unload_params('r', 'alpha', 's', 'beta')

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


def _fit_and_simulate(model, parameters, t):
    data = model.generateData(t=t, size=100, parameters=model.parameters_dictionary_from_list(parameters))
    model.fit(data['frequency'], data['recency'], data['T'])
    return model
