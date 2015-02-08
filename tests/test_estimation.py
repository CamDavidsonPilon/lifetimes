from __future__ import print_function

import numpy as np
import pandas as pd

import numpy.testing as npt
import pytest

import lifetimes.estimation as estimation

@pytest.fixture()
def cdnow_customers():
    return pd.read_csv('lifetimes/datasets/cdnow_customers.csv', sep='\s+', index_col=[0])



class TestBetaGammaFitter():

    def test_sum_of_scalar_inputs_to_negative_log_likelihood_is_equal_to_array(self):
        bgf = estimation.BetaGeoFitter
        x = np.array([1,3])
        t_x = np.array([2,2])
        t = np.array([5,6])
        params = [1,1,1,1]
        assert bgf._negative_log_likelihood(params, x[0], t_x[0], t[0]) + bgf._negative_log_likelihood(params, x[1], t_x[1], t[1]) \
                == bgf._negative_log_likelihood(params, x, t_x, t)
 
    def test_params_out_is_close_to_Hardie_paper(self, cdnow_customers):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['x'], cdnow_customers['t_x'], cdnow_customers['T'])
        expected = np.array([0.243, 4.414, 0.793, 2.426])
        npt.assert_array_almost_equal(expected, np.array(bfg._unload_params()), decimal=3)

    def test_conditional_expectation_returns_same_value_as_Hardie_excel_sheeet(self, cdnow_customers):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['x'], cdnow_customers['t_x'], cdnow_customers['T'])
        x = 2
        t_x = 30.43
        T = 38.86
        t = 39 
        expected = 1.226
        actual = bfg.conditional_expected_number_of_purchases_up_to_time(t, x, t_x, T) 
        assert abs(expected - actual) < 0.001

    def test_expecatation_returns_same_value_minus_one_as_Hardie_excel_sheet(self, cdnow_customers):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['x'], cdnow_customers['t_x'], cdnow_customers['T'])

        times = np.array([0.1429, 1.0, 3.00, 31.8571, 32.00, 78.00])
        expected = np.array([0.0078 ,0.0532 ,0.1506 ,1.0405,1.0437, 1.8576]) + 1
        actual = bfg.expected_number_of_purchases_up_to_time(times)
        npt.assert_array_almost_equal(actual, expected, decimal=3) 

    def test_conditional_probability_alive_returns_1_if_no_repeat_purchases(self, cdnow_customers):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['x'], cdnow_customers['t_x'], cdnow_customers['T'])

        assert bfg.conditional_probability_alive(0, 1, 1) == 1.0

