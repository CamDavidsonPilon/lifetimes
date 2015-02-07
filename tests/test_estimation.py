import numpy as np
import pandas as pd

import numpy.testing as npt
import pytest

import lifetimes.estimation as estimation

@pytest.fixture()
def cdnow_customers():
    return pd.read_csv('lifetimes/datasets/cdnow_customers.csv', sep='\s+', index_col=[0])



class TestBGNBDFitter():


    def test_sum_of_scalar_inputs_to_negative_log_likelihood_is_equal_to_array(self):
        bgf = estimation.BGNBDFitter
        x = np.array([1,3])
        t_x = np.array([2,2])
        t = np.array([5,6])
        params = [1,1,1,1]
        assert bgf._negative_log_likelihood(params, x[0], t_x[0], t[0]) + bgf._negative_log_likelihood(params, x[1], t_x[1], t[1]) \
                == bgf._negative_log_likelihood(params, x, t_x, t)
 
    def test_params_out_is_close_to_Hardie_paper(self, cdnow_customers):
        bfg = estimation.BGNBDFitter()
        bfg.fit(cdnow_customers['x'], cdnow_customers['t_x'], cdnow_customers['T'])
        expected = np.array([0.243, 4.414, 0.793, 2.426])
        npt.assert_array_almost_equal(expected, np.array(bfg._unload_params()), decimal=3)
