from __future__ import generator_stop
from __future__ import annotations

import os
from abc import ABC
import warnings
import inspect

import pytest

import pandas as pd
import numpy as np
import arviz as az

import pymc as pm
import aesara.tensor as at

import btyd
import btyd.utils as utils
from btyd.datasets import (
    load_cdnow_summary,
    load_cdnow_summary_data_with_monetary_value,
    load_donations,
    load_transaction_data,
)


PATH_BGNBD_MODEL = "./bgnbd.json"


class TestBetaGeoModel:

    @pytest.fixture(scope='class')
    def fitted_bgm(self,cdnow_customers): # ADD TYPE HINTING
        """ For running multiple tests on a single BetaGeoModel fit() instance. """

        bgm = btyd.BetaGeoModel().fit(cdnow_customers)
        return bgm

    def test_hyperparams(self):
        """
        GIVEN an uninstantiated BetaGeoModel,
        WHEN it is instantiated with custom values for hyperparams,
        THEN BetaGeoModel._hyperparams should differ from defaults and match the custom values set by the user. 
        """
    
        custom_hyperparams = {
            'alpha_prior_alpha': 2.,
            'alpha_prior_beta': 4.,
            'r_prior_alpha': 3.,
            'r_prior_beta': 2.,
            'phi_prior_lower': .1,
            'phi_prior_upper': .99,
            'kappa_prior_alpha': 1.1,
            'kappa_prior_m': 2.5,
        }

        default_bgm = btyd.BetaGeoModel()
        custom_bgm = btyd.BetaGeoModel(hyperparams = custom_hyperparams)

        assert default_bgm._hyperparams != custom_bgm._hyperparams
        assert custom_hyperparams == custom_bgm._hyperparams

    def test_log_likelihood(self):
        """
        GIVEN the BetaGeo log-likelihood function,
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
        loglike_terms = btyd.BetaGeoModel._log_likelihood(self,**values,testing=True)
        expected = np.array([854.424,-748.1218,9e-05,3.97e-03])
        np.testing.assert_allclose(loglike_terms,expected,rtol=1e-04)

        # Test output.
        loglike_out = btyd.BetaGeoModel._log_likelihood(self,**values).eval()
        expected = np.array([100.7957])
        np.testing.assert_allclose(loglike_out,expected,rtol=1e-04)

    def test_repr(self,fitted_bgm):
        """
        GIVEN a declared BetaGeo concrete class object,
        WHEN the string representation is called on this object,
        THEN string representations of library name, module, BetaGeoModel class, parameters, and # rows used in estimation are returned.
        """

        assert repr(btyd.BetaGeoModel) == "<class 'btyd.models.beta_geo_model.BetaGeoModel'>"
        assert repr(btyd.BetaGeoModel()) == "<btyd.BetaGeoModel>"
        
        # Expected parameters may vary slightly due to rounding errors.
        expected = [
             "<btyd.BetaGeoModel: Parameters {'alpha': 4.4, 'r': 0.2, 'a': 0.8, 'b': 2.4} estimated with 2357 customers.>",
              "<btyd.BetaGeoModel: Parameters {'alpha': 4.5, 'r': 0.2, 'a': 0.8, 'b': 2.4} estimated with 2357 customers.>",
        ]
        assert any(expected) == True
    
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
        summary = az.summary(
            data=fitted_bgm.idata, 
            var_names=['BetaGeoModel::a','BetaGeoModel::b','BetaGeoModel::alpha','BetaGeoModel::r']
            )
        assert isinstance(summary,pd.DataFrame)
    
    def test_unload_params(self, fitted_bgm):
        """
        GIVEN a Bayesian BetaGeoModel fitted on the CDNOW dataset,
        WHEN its parameters are checked via self._unload_params(),
        THEN they should be within 1e-01 tolerance of the MLE parameters from the original paper.
        """

        expected = np.array([4.414, 0.243, 0.793, 2.426])
        np.testing.assert_allclose(expected, np.array(fitted_bgm._unload_params()),rtol=1e-01)
    
    def test_posterior_sampling(self,fitted_bgm):
        """
        GIVEN a Bayesian BetaGeoModel fitted on the CDNOW dataset,
        WHEN its posterior parameter distributions are sampled via self._unload_params(posterior = True),
        THEN the length of the numpy arrays should match that of the n_samples argument.
        """

        sampled_posterior_params = fitted_bgm._unload_params(posterior=True)

        assert len(sampled_posterior_params) == 4
        assert sampled_posterior_params[0].shape == (100,)


    def test_conditional_expected_number_of_purchases_up_to_time(self, fitted_bgm):
        """
        GIVEN a Bayesian BetaGeoModel fitted on the CDNOW dataset,
        WHEN self._conditional_expected_number_of_purchases_up_to_time() is called,
        THEN it should return a value within 1e-02 tolerance to the expected MLE output from the original paper.
        """

        fitted_bgm._frequency = 2
        fitted_bgm._recency = 30.43
        fitted_bgm._T = 38.86
        t = 39

        expected = np.array(1.226)
        actual = fitted_bgm._conditional_expected_number_of_purchases_up_to_time(t)
        np.testing.assert_allclose(expected, actual,rtol=1e-02)

    def test_expected_number_of_purchases_up_to_time(self, fitted_bgm):
        """
        GIVEN a Bayesian BetaGeoModel fitted on the CDNOW dataset,
        WHEN self._expected_number_of_purchases_up_to_time() is called,
        THEN it should return a value within 1e-02 tolerance to the expected MLE output from the original paper.
        """

        times = np.array([0.1429, 1.0, 3.00, 31.8571, 32.00, 78.00])
        expected = np.array([0.0078, 0.0532, 0.1506, 1.0405, 1.0437, 1.8576])
        actual = fitted_bgm._expected_number_of_purchases_up_to_time(times, None)
        np.testing.assert_allclose(actual,expected,rtol=1e-02)

    def test_conditional_probability_alive(self, fitted_bgm):
        """
        GIVEN a fitted BetaGeoModel object,
        WHEN self._conditional_probability_alive() is called,
        THEN output should always be between 0 and 1, and 1 if a customer has only made a single purchase.
        """

        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                for k in range(j, 100, 10):
                    assert 0 <= fitted_bgm._conditional_probability_alive(None, None, False, 100, i, j, k) <= 1.0
        assert fitted_bgm._conditional_probability_alive(None, None, False, 100, 0, 1, 1) == 1.0

    def test_probability_of_n_purchases_up_to_time(self,fitted_bgm):
        """ 
        GIVEN a fitted BetaGeoModel object,
        WHEN self._probability_of_n_purchases_up_to_time() is called,
        THEN output should approximate that of the BTYD R package: https://cran.r-project.org/web/packages/BTYD/BTYD.pdf 
        """

        # probability that a customer will make 10 repeat transactions in the
        # time interval (0,2]
        expected = np.array(1.07869e-07)
        actual = fitted_bgm._probability_of_n_purchases_up_to_time(2, 10)
        np.testing.assert_allclose(expected, actual,rtol=1e-01)

        # probability that a customer will make no repeat transactions in the
        # time interval (0,39]
        expected = 0.5737864
        actual = fitted_bgm._probability_of_n_purchases_up_to_time(39, 0)
        np.testing.assert_allclose(expected, actual,rtol=1e-03)

        # PMF
        expected = np.array(
            [
                0.0019995214,
                0.0015170236,
                0.0011633150,
                0.0009003148,
                0.0007023638,
                0.0005517902,
                0.0004361913,
                0.0003467171,
                0.0002769613,
                0.0002222260,
            ]
        )
        actual = np.array([fitted_bgm._probability_of_n_purchases_up_to_time(30, n) for n in range(11, 21)]).flatten()
        np.testing.assert_allclose(expected, actual,rtol=1e-02)
    
    def test_save_model(self, fitted_bgm):
        """
        GIVEN a fitted BetaGeoModel object,
        WHEN self.save_model() is called,
        THEN the external file should exist.
        """

        # os.remove(PATH_BGNBD_MODEL)
        assert os.path.isfile(PATH_BGNBD_MODEL) == False
        
        fitted_bgm.save_model(PATH_BGNBD_MODEL)
        assert os.path.isfile(PATH_BGNBD_MODEL) == True

    def test_load_predict(self, fitted_bgm):
        """
        GIVEN fitted and unfitted BetaGeoModel objects,
        WHEN parameters of the fitted model are loaded from an external JSON via self.load_model(),
        THEN InferenceData unloaded parameters should match, raising exceptions otherwise and if predictions attempted without RFM data.
        """

        bgm_new = btyd.BetaGeoModel()
        bgm_new.load_model(PATH_BGNBD_MODEL)
        assert isinstance(bgm_new.idata,az.InferenceData)
        assert bgm_new._unload_params() == fitted_bgm._unload_params()
        
        # assert param exception (need another saved model and additions to self.load_model())
        # assert prediction exception

        os.remove(PATH_BGNBD_MODEL)
    
    def test_quantities_of_interest(self):
        """
        GIVEN the _quantities_of_interest BaseModel attribute,
        WHEN the keys of the '_quantities_of_interest' call dictionary attribute are called,
        THEN they should match the list of expected keys.
        """

        expected = ['cond_prob_alive', 'cond_n_prchs_to_time', 'n_prchs_to_time', 'prob_n_prchs_to_time']
        actual = list(btyd.BetaGeoModel._quantities_of_interest.keys()) 
        assert actual == expected
    
    @pytest.mark.parametrize("qoi, instance",[("cond_prob_alive",np.ndarray),
                                            ("cond_n_prchs_to_time",np.float64), 
                                            ("n_prchs_to_time",np.float64),
                                            ("prob_n_prchs_to_time",np.ndarray)])
    def test_predict_mean(self,fitted_bgm,cdnow_customers, qoi, instance):
        """
        GIVEN a fitted BetaGeoModel,
        WHEN all four quantities of interest are called via BetaGeoModel.predict() for posterior mean predictions,
        THEN expected output instances and dimensions should be returned.
        """

        array_out = fitted_bgm.predict(method=qoi,rfm_df=cdnow_customers,t=10,n=5)

        assert isinstance(array_out,instance)
    
    # TODO: Add a param to test posterior draws
    @pytest.mark.parametrize("qoi, instance, draws",[("cond_prob_alive",np.ndarray, 100),
                                        ("cond_n_prchs_to_time",np.ndarray, 200), 
                                        ("n_prchs_to_time",np.ndarray, 300),
                                        ("prob_n_prchs_to_time",np.ndarray, 400)])
    def test_predict_full(self,fitted_bgm,cdnow_customers,qoi, instance, draws):
        """
        GIVEN a fitted BetaGeoModel,
        WHEN all four quantities of interest are called via BetaGeoModel.predict() for full posterior predictions,
        THEN expected output instances and dimensions should be returned.
        """

        array_out = fitted_bgm.predict(method=qoi,rfm_df=cdnow_customers,t=10,n=5,sample_posterior=True, posterior_draws=draws)

        assert isinstance(array_out,instance)
        assert len(array_out) == draws

    def test_generate_rfm_data(self, fitted_bgm):
        """
        GIVEN a fitted BetaGeoModel,
        WHEN synthetic data is generated from its parameters,
        THEN the resultant dataframe should contain the expected column names and row count.
        """

        # Test default value of size argument.
        synthetic_df = fitted_bgm.generate_rfm_data()
        assert len(synthetic_df) == 1000

        # Test custom value of size argument.
        synthetic_df = fitted_bgm.generate_rfm_data(size=123)
        assert len(synthetic_df) == 123

        expected_cols = ["frequency", "recency", "T", "lambda", "p", "alive"]
        actual_cols = list(synthetic_df.columns)

        assert actual_cols == expected_cols    
