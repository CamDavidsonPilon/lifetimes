from __future__ import generator_stop
from __future__ import annotations

import os
import warnings

import pytest

import pandas as pd
import numpy as np
# from numpy import array, isin
import arviz as az

import pymc as pm
import aesara.tensor as at

import btyd as lt
import btyd.utils as utils
from btyd.datasets import (
    load_cdnow_summary,
    load_cdnow_summary_data_with_monetary_value,
    load_donations,
    load_transaction_data,
)


PATH_BGNBD_MODEL = "./bgnbd.json"

@pytest.fixture(scope='module')
def cdnow_customers():
    """ Create an RFM dataframe for multiple tests and fixtures. """
    rfm_df = load_cdnow_summary_data_with_monetary_value()
    return rfm_df


class TestBaseModel:

    def test_repr(self):
        """
        GIVEN a BaseModel that has not been instantiated,
        WHEN repr() is called on this object,
        THEN a string representation containing library name, module and model class are returned.
        """
        
        assert repr(lt.BaseModel) == "<class 'btyd.models.BaseModel'>"

    def test_sample(self):
        """
        GIVEN the _sample() static method,
        WHEN a numpy array and sample quantity are provided,
        THEN a numpy array of the specified length containing some or all of the original elements is returned.
        """
        posterior_distribution = np.array([.456,.358,1.8,2.,.999])
        samples = 7
        posterior_samples = lt.BaseModel._sample(posterior_distribution,samples) 
        assert len(posterior_samples) == samples
        
        # Convert numpy arrays to sets to test intersections of elements.
        dist_set = set(posterior_distribution.flatten())
        sample_set = set(posterior_samples.flatten())
        assert len(sample_set.intersection(dist_set)) <= len(posterior_distribution)
    
    def test_dataframe_parser(self,cdnow_customers):
        """
        GIVEN an RFM dataframe,
        WHEN the _dataframe_parser() static method is called on it,
        THEN five numpy arrays should be returned.
        """

        parsed = lt.BaseModel._dataframe_parser(cdnow_customers) 
        assert len(parsed) == 5


class TestBetaGeoModel:

    @pytest.fixture(scope='class')
    def fitted_bgm(self,cdnow_customers):
        """ For running multiple tests on a single BetaGeoModel fit() instance. """

        bgm = lt.BetaGeoModel().fit(cdnow_customers)
        return bgm
    
    def test_loglike(self):
        """
        GIVEN the BetaGeo log-likelihood function
        WHEN it is called with the inputs and parameters specified in Farder-Hardie's notes,
        THEN term values and output should match those in the paper. 
        """
        
        values = {
            'frequency':200,
            'recency':38,
            'T': 40,
            'r': 0.25,
            'alpha': 4.,
            'a': 0.8,
            'b': 2.5
        }

        # Test term values.
        loglike_terms = lt.BetaGeoModel._loglike(self,**values,testing=True)
        expected = np.array([854.424,-748.1218,9e-05,3.97e-03])
        np.testing.assert_allclose(loglike_terms,expected,rtol=1e-04)

        # Test output.
        loglike_out = lt.BetaGeoModel._loglike(self,**values).eval()
        expected = np.array([100.7957])
        np.testing.assert_allclose(loglike_out,expected,rtol=1e-04)

    def test_repr(self,fitted_bgm):
        """
        GIVEN a declared BetaGeo concrete class object,
        WHEN the string representation is called on this object,
        THEN string representations of library name, module, BetaGeoModel class, parameters, and # rows used in estimation are returned.
        """

        assert repr(lt.BetaGeoModel) == "<class 'btyd.models.beta_geo_model.BetaGeoModel'>"
        assert repr(lt.BetaGeoModel()) == "<btyd.BetaGeoModel>"
        assert repr(fitted_bgm) == "<btyd.BetaGeoModel: Parameters {'alpha': array([4.4]), 'r': array([0.2]), 'a': array([0.8]), 'b': array([2.4])} estimated with 2357 customers.>"
    
    def test_model(self,fitted_bgm):
        """
        GIVEN an instantiated BetaGeo model,
        WHEN _model is called,
        THEN it should contain the specified random variables.
        """

        model = fitted_bgm._model()
        expected = '[BetaGeoModel::alpha, BetaGeoModel::r, BetaGeoModel::phi, BetaGeoModel::kappa, BetaGeoModel::a, BetaGeoModel::b]'
        assert str(model.unobserved_RVs) == expected
    
    def test_fit(self,fitted_bgm):
        """
        GIVEN a BetaGeoModel() object,
        WHEN it is fitted,
        THEN the new instantiated attributes should include an arviz InferenceData class and dict with required model parameters.
        """

        assert isinstance(fitted_bgm.idata,az.InferenceData)

        # Check if arviz methods are supported.
        summary = az.summary(data=fitted_bgm.idata, var_names=['BetaGeoModel::a','BetaGeoModel::b','BetaGeoModel::alpha','BetaGeoModel::r'])
        assert isinstance(summary,pd.DataFrame)
    
    def test_unload_params(self, fitted_bgm):
        """
        GIVEN a Bayesian BetaGeoModel fitted on the CDNOW dataset,
        WHEN its parameters are checked via self._unload_params()
        THEN they should be within 1e-01 tolerance of the MLE parameters from the original paper.
        """

        expected = np.array([[4.414], [0.243], [0.793], [2.426]])
        np.testing.assert_allclose(expected, np.array(fitted_bgm._unload_params()),rtol=1e-01)

    def test_conditional_expected_number_of_purchases_up_to_time(self, fitted_bgm):
        """
        GIVEN a Bayesian BetaGeoModel fitted on the CDNOW dataset,
        WHEN self.conditional_expected_number_of_purchases_up_to_time() is called,
        THEN it should return a value within 1e-02 tolerance to the expected MLE output from the original paper.
        """

        x = 2
        t_x = 30.43
        T = 38.86
        t = 39
        expected = np.array(1.226)
        actual = fitted_bgm.conditional_expected_number_of_purchases_up_to_time(t, x, t_x, T)
        np.testing.assert_allclose(expected, actual,rtol=1e-02)

    def test_expected_number_of_purchases_up_to_time(self, fitted_bgm):
        """
        GIVEN a Bayesian BetaGeoModel fitted on the CDNOW dataset,
        WHEN self.expected_number_of_purchases_up_to_time() is called,
        THEN it should return a value within 1e-02 tolerance to the expected MLE output from the original paper.
        """

        times = np.array([0.1429, 1.0, 3.00, 31.8571, 32.00, 78.00])
        expected = np.array([0.0078, 0.0532, 0.1506, 1.0405, 1.0437, 1.8576])
        actual = fitted_bgm.expected_number_of_purchases_up_to_time(times)
        np.testing.assert_allclose(actual,expected,rtol=1e-02)

    def test_conditional_probability_alive(self, fitted_bgm):
        """
        GIVEN a fitted BetaGeoModel object,
        WHEN self.conditional_probability_alive() is called,
        THEN output should always be between 0 and 1, and 1 if a customer has only made a single purchase.
        """

        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                for k in range(j, 100, 10):
                    assert 0 <= fitted_bgm.conditional_probability_alive(i, j, k) <= 1.0
        assert fitted_bgm.conditional_probability_alive(0, 1, 1) == 1.0

    def test_probability_of_n_purchases_up_to_time(self,fitted_bgm):
        """ 
        GIVEN a fitted BetaGeoModel object,
        WHEN self.probability_of_n_purchases_up_to_time() is called,
        THEN output should approximate that of the BTYD R package: https://cran.r-project.org/web/packages/BTYD/BTYD.pdf 
        """

        # probability that a customer will make 10 repeat transactions in the
        # time interval (0,2]
        expected = np.array(1.07869e-07)
        actual = fitted_bgm.probability_of_n_purchases_up_to_time(2, 10)
        np.testing.assert_allclose(expected, actual,rtol=1e-01)

        # probability that a customer will make no repeat transactions in the
        # time interval (0,39]
        expected = 0.5737864
        actual = fitted_bgm.probability_of_n_purchases_up_to_time(39, 0)
        np.testing.assert_allclose(expected, actual,rtol=1e-03)

        # PMF
        expected = np.array(
            [
                [0.0019995214],
                [0.0015170236],
                [0.0011633150],
                [0.0009003148],
                [0.0007023638],
                [0.0005517902],
                [0.0004361913],
                [0.0003467171],
                [0.0002769613],
                [0.0002222260],
            ]
        )
        actual = np.array([fitted_bgm.probability_of_n_purchases_up_to_time(30, n) for n in range(11, 21)])
        np.testing.assert_allclose(expected, actual,rtol=1e-02)
    
    def test_save_params(self, fitted_bgm):
        """
        GIVEN a fitted BetaGeoModel object,
        WHEN self.save_model() is called,
        THEN the external file should exist.
        """

        # os.remove(PATH_BGNBD_MODEL)
        assert os.path.isfile(PATH_BGNBD_MODEL) == False
        
        fitted_bgm.save_params(PATH_BGNBD_MODEL)
        assert os.path.isfile(PATH_BGNBD_MODEL) == True

    def test_load_predict(self, cdnow_customers, fitted_bgm):
        """
        GIVEN fitted and unfitted BetaGeoModel objects,
        WHEN parameters of the fitted model are loaded via self.load_params() and self.predict() is called on both models,
        THEN parameters and predictions should match for both, raising exceptions otherwise and for predictions attempted without RFM data.
        """

        bgm_new = lt.BetaGeoModel()
        bgm_new.load_params(PATH_BGNBD_MODEL)
        assert bgm_new._unload_params() == fitted_bgm._unload_params()
        
        # assert param exception (need another saved model and additions to self.load_params())
        # assert predictions match (@pytest.parameterize()?)
        # assert prediction exception

        os.remove(PATH_BGNBD_MODEL)
    
    def test_generate_rfm_data(self, cdnow_customers, fitted_bgm):
        """
        GIVEN fitted BetaGeoModel and BetaGeoFitter objects,
        WHEN synthetic data is generated from their parameters,
        THEN the BetaGeoModel should have a better posterior predictive deviation metric.
        """

        # 
        frequency, recency, T, _, _ = lt.BaseModel._dataframe_parser(cdnow_customers) 
        mle_df = lt.BetaGeoFitter().fit(frequency, recency, T).generate_new_data(size=len(cdnow_customers))
        mle_freq = mle_df['frequency'].values
        mle_rec = mle_df['recency'].values
        ppc_mle = utils.posterior_predictive_deviation(frequency, mle_freq, recency, mle_rec)

        bayes_df = fitted_bgm.generate_rfm_data()
        bayes_freq = mle_df['frequency'].values
        bayes_rec = mle_df['recency'].values
        ppc_bayes_self = utils.posterior_predictive_deviation(fitted_bgm.frequency, bayes_freq, fitted_bgm.recency, bayes_rec)
        ppc_bayes_obs = utils.posterior_predictive_deviation(frequency, bayes_freq, recency, bayes_rec)
        assert ppc_bayes_self == ppc_bayes_obs

        assert ppc_bayes_obs <= ppc_mle #0.0899339600992113!, # 0.12174678527333853!

        

    