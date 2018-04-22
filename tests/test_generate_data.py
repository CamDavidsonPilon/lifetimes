from collections import OrderedDict
import pytest
import pandas as pd
import numpy as np
import numpy.testing as npt

import scipy.stats as stats

import lifetimes.estimation as estimation
from lifetimes.generate_data import beta_geometric_nbd_model, pareto_nbd_model, modified_beta_geometric_nbd_model, \
    beta_geometric_beta_binom_model


class TestBetaGeoGeneration():
    params = [0.243, 4.414, 0.793, 2.426]


class TestParetoNBDGeneration():
    params = [0.553, 10.578, 0.606, 11.669]


class TestModifiedBetaGeoNBDGeneration():
    params = [0.525, 6.183, 0.891, 1.614]
    times = np.array([0.1429, 1.0, 3.00, 31.8571, 32.00, 78.00])
    expected = np.array([0.0078, 0.0532, 0.1506, 1.0405, 1.0437, 1.8576])


class TestBetaGeoBetaBinomGeneration():

    def __init__(self):
        self.params = OrderedDict([('alpha', 1.204), ('beta', 0.750), ('gamma', 0.657),
                                   ('delta', 2.783)])
        np.random.seed(188898)

    def test_positivity(self):
        sim_data = beta_geometric_beta_binom_model(N=6, **self.params, size=5000)
        assert (sim_data['frequency'] >= 0).all()
        assert (sim_data['recency'] >= 0).all()

    def test_hitting_max(self):
        sim_data = beta_geometric_beta_binom_model(N=6, **self.params, size=5000)
        assert sim_data['frequency'].max() == 6
        assert sim_data['recency'].max() == 6

    def test_alive_probs(self):     
        sim_data = beta_geometric_beta_binom_model(N=6, **self.params, size=50000)
        assert (np.abs(sim_data.loc[(sim_data['frequency'] == 0) & (sim_data['recency'] == 0),
                                    'alive'].mean() - 0.11) < 0.01)
        assert (np.abs(sim_data.loc[(sim_data['frequency'] == 2) & (sim_data['recency'] == 4),
                                    'alive'].mean() - 0.59) < 0.01)
        assert (np.abs(sim_data.loc[(sim_data['frequency'] == 6) & (sim_data['recency'] == 6),
                                    'alive'].mean() - 0.93) < 0.01)

    def test_params_same_from_sim_data(self):
        sim_data = beta_geometric_beta_binom_model(N=6, **self.params, size=100000)
        bbtf = estimation.BetaGeoBetaBinomFitter()
        grouped_data = sim_data.groupby(['frequency', 'recency', 'n'])['customer_id'].count()
        grouped_data = grouped_data.reset_index().rename(columns={'customer_id': 'n_custs'})
        bbtf.fit(grouped_data['frequency'],
                 grouped_data['recency'],
                 grouped_data['n'],
                 grouped_data['n_custs'])
        assert ((np.array(list(self.params.values())) - np.array(
            bbtf._unload_params('alpha', 'beta', 'gamma', 'delta'))) < 0.1).all()
