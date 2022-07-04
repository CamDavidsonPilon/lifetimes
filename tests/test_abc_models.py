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


@pytest.fixture(scope='module')
def cdnow_customers() -> pd.DataFrame:
    """ Create an RFM dataframe for multiple tests and fixtures. """
    rfm_df = load_cdnow_summary_data_with_monetary_value()
    return rfm_df

@pytest.mark.parametrize("obj",[btyd.BaseModel, btyd.AliveMixin])
def test_isabstract(obj):
        """
        GIVEN the BaseModel and AliveMixin model factory objects,
        WHEN they are inspected for inheritance from ABC,
        THEN they should both identify as abstract objects.
        """

        assert inspect.isabstract(obj) is True


class TestBaseModel:

    def test_repr(self):
        """
        GIVEN a BaseModel that has not been instantiated,
        WHEN repr() is called on this object,
        THEN a string representation containing library name, module and model class are returned.
        """
        
        assert repr(btyd.BaseModel) == "<class 'btyd.models.BaseModel'>"
    
    def test_abstract_methods(self):
        """
        GIVEN the BaseModel model factory object,
        WHEN its abstract methods are overridden,
        THEN they should all return None.
        """

        # Override abstract methods:
        btyd.BaseModel.__abstractmethods__ = set()

        # Create concrete class for testing:
        class ConcreteBaseModel(btyd.BaseModel):
            pass
        
        # Instantiate concrete testing class and call all abstrast methods:
        concrete_base = ConcreteBaseModel()
        model = concrete_base._model()
        log_likelihood = concrete_base._log_likelihood()
        generate_rfm_data = concrete_base.generate_rfm_data()

        assert model is None
        assert log_likelihood is None
        assert generate_rfm_data is None

    def test_sample(self):
        """
        GIVEN the _sample() static method,
        WHEN a numpy array and sample quantity are provided,
        THEN a numpy array of the specified length containing some or all of the original elements is returned.
        """
        posterior_distribution = np.array([.456,.358,1.8,2.,.999])
        samples = 7
        posterior_samples = btyd.BaseModel._sample(posterior_distribution,samples) 
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

        parsed = btyd.BaseModel._dataframe_parser(cdnow_customers) 
        assert len(parsed) == 5


class TestAliveMixin:
    
    def test_call_dict(self):
        """
        GIVEN the AliveMixin model factory object,
        WHEN the keys of the 'quantities_of_interest' call dictionary attribute are returned,
        THEN they should match the list of expected keys.
        """

        expected = ['cond_prob_alive', 'cond_n_prchs_to_time', 'n_prchs_to_time', 'prob_n_prchs_to_time']
        actual = list(btyd.AliveMixin.quantities_of_interest.keys()) 
        assert actual == expected
    
    def test_abstract_methods(self):
        """
        GIVEN the AliveMixin model factory object,
        WHEN its abstract methods are overridden,
        THEN they should all return None.
        """

        # Override abstract methods:
        btyd.AliveMixin.__abstractmethods__ = set()

        # Create concrete class for testing:
        class ConcreteAliveMixin(btyd.AliveMixin):
            pass
        
        # Instantiate concrete testing class and call all abstrast methods:
        concrete_api = ConcreteAliveMixin()
        cond_prob_alive = concrete_api._conditional_probability_alive()
        cond_n_prchs_to_time = concrete_api._conditional_expected_number_of_purchases_up_to_time()
        n_prchs_to_time = concrete_api._expected_number_of_purchases_up_to_time()
        prob_n_prchs_to_time = concrete_api._probability_of_n_purchases_up_to_time()

        assert cond_prob_alive is None
        assert cond_n_prchs_to_time is None
        assert cond_prob_alive is None
        assert n_prchs_to_time is None
