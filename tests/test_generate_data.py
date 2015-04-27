import pytest 
import pandas as pd
import numpy as np
import scipy.stats as stats

from lifetimes.generate_data import beta_geometric_nbd_model, beta_geometric_nbd_model2


def test_statistics_of_beta_geometric_nbd_model():
    # This test is terrible, and should be based on actual statistics.
    N = 10000
    r, alpha, a, b = 0.11, 0.05, 1.41, 0.56
    df = beta_geometric_nbd_model2(350*np.ones(N), r, alpha, a, b, N)
    assert abs(df['frequency'].mean() - 1.26) < 0.2
    assert abs(df['lambda'].mean() - 2.272629382453732) < 0.2
    assert abs(df['p'].mean() - 0.7159928181941454) < 0.1
    assert abs(df['alive'].mean() - 0.3845) < 0.05



