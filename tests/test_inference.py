from __future__ import generator_stop
from __future__ import annotations

import os
import warnings

import pytest

import pandas as pd
import numpy as np

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


class TestBaseInference:
    def test_repr(self):
        """Test string representation of BaseInference class."""
        base_infrnc = lt.BaseInference()
        assert repr(base_infrnc) == "<btyd.BaseInference>"
        base_infrnc.data = np.array([1, 2, 3])
        assert repr(base_infrnc) == "<btyd.BaseInference: Estimated with 3 samples.>"
