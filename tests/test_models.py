from __future__ import generator_stop
from __future__ import annotations

import os
import warnings

import pytest

import pandas as pd
import numpy as np
from numpy import array, isin

import pymc as pm
import aesara.tensor as at

import lifetimes as lt
import lifetimes.utils as utils
from lifetimes.datasets import (
    load_cdnow_summary,
    load_cdnow_summary_data_with_monetary_value,
    load_donations,
    load_transaction_data,
)

PATH_SAVE_MODEL = "./base_inference.pkl"


@pytest.fixture
def cdnow_customers():
    return load_cdnow_summary()


class TestBaseInferencer:
    def test_repr(self):
        """
        GIVEN a declared BaseInferencer abstract class object,
        WHEN the string representation is called on this object,
        THEN string representations of model name, rows/subjects used in estimation, and parameters are returned.
        """
        assert repr(lt.BaseInferencer) == "<class 'lifetimes.models.BaseInferencer'>"
    # Can only test static method from this class.
    # test to assert method class ownership?
    def test_sample(self):
        """
        GIVEN the _sample() static method,
        WHEN a numpy array and sample quantity are specified,
        THEN a numpy array of the specified length containing some or all of the original elements is returned.
        """
        posterior_distribution = array([.456,.358,1.8,2.,.999])
        samples = 7
        posterior_samples = lt.BaseInferencer._sample(posterior_distribution,samples) 
        assert len(posterior_samples) == samples
        
        # Convert numpy arrays to sets to test intersections of elements.
        dist_set = set(posterior_distribution.flatten())
        sample_set = set(posterior_samples.flatten())
        assert len(sample_set.intersection(dist_set)) < len(posterior_distribution)

class TestBetaGeoInference:
    @pytest.fixture()
    def donations(self):
        return load_donations()
    
    def test_repr(self):
        """
        GIVEN a declared BetaGeo concrete class object,
        WHEN the string representation is called on this object,
        THEN return string representations of model name, rows/subjects used in estimation, contingent on data & param attributes.
        """
        base_infrnc = lt.BaseInferencer()
        assert repr(base_infrnc) == "<btyd.BaseInferencer>"
        base_infrnc.data = array([1, 2, 3])
        base_infrnc.param_str = ', '.join(str(p) for p in ['a','b','alpha','r'])
        # assert repr(base_infrnc) == "<btyd.BaseInferencer: a, b, alpha, r posterior parameters estimated with 3 subjects.>"
    