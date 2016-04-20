import pytest
import pandas as pd
import numpy as np
import scipy.stats as stats

import lifetimes.generate_data as gen

from lifetimes import models

@pytest.mark.model_fitting
def test_model_fitting():

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
    print model.numerical_metrics

    model.simulate()

    print "After simulating"
    print model.numerical_metrics

def test_model_simulation():
    assert True