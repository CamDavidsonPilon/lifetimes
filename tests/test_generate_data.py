from collections import OrderedDict
import pytest
import pandas as pd
import numpy as np
import numpy.testing as npt

import scipy.stats as stats

import lifetimes.estimation as estimation
from lifetimes.generate_data import beta_geometric_nbd_model, pareto_nbd_model, modified_beta_geometric_nbd_model, \
    beta_geometric_beta_binom_model, beta_geometric_nbd_model_transactional_data
from lifetimes.utils import summary_data_from_transaction_data


def setup_module(module):
    np.random.seed(188898)


class TestBetaGeoGeneration():
    params = [0.243, 4.414, 0.793, 2.426]


class TestParetoNBDGeneration():
    params = [0.553, 10.578, 0.606, 11.669]


class TestModifiedBetaGeoNBDGeneration():
    params = [0.525, 6.183, 0.891, 1.614]
    times = np.array([0.1429, 1.0, 3.00, 31.8571, 32.00, 78.00])
    expected = np.array([0.0078, 0.0532, 0.1506, 1.0405, 1.0437, 1.8576])





class TestBetaGeoBetaBinomGeneration():

    @pytest.fixture()
    def bbgb_params(self):
        return OrderedDict([('alpha', 1.204), ('beta', 0.750), ('gamma', 0.657), ('delta', 2.783)])

    def test_positivity(self, bbgb_params):
        sim_data = beta_geometric_beta_binom_model(N=6, size=5000, **bbgb_params)
        assert (sim_data['frequency'] >= 0).all()
        assert (sim_data['recency'] >= 0).all()

    def test_hitting_max(self, bbgb_params):
        sim_data = beta_geometric_beta_binom_model(N=6, size=5000, **bbgb_params)
        assert sim_data['frequency'].max() == 6
        assert sim_data['recency'].max() == 6

    def test_alive_probs(self, bbgb_params):
        sim_data = beta_geometric_beta_binom_model(N=6, size=50000, **bbgb_params)
        assert (np.abs(sim_data.loc[(sim_data['frequency'] == 0) & (sim_data['recency'] == 0),
                                    'alive'].mean() - 0.11) < 0.01)
        assert (np.abs(sim_data.loc[(sim_data['frequency'] == 2) & (sim_data['recency'] == 4),
                                    'alive'].mean() - 0.59) < 0.01)
        assert (np.abs(sim_data.loc[(sim_data['frequency'] == 6) & (sim_data['recency'] == 6),
                                    'alive'].mean() - 0.93) < 0.01)

    def test_params_same_from_sim_data(self, bbgb_params):
        sim_data = beta_geometric_beta_binom_model(N=6, size=100000, **bbgb_params)
        bbtf = estimation.BetaGeoBetaBinomFitter()
        grouped_data = sim_data.groupby(['frequency', 'recency', 'n_periods'])['customer_id'].count()
        grouped_data = grouped_data.reset_index().rename(columns={'customer_id': 'weights'})
        bbtf.fit(grouped_data['frequency'],
                 grouped_data['recency'],
                 grouped_data['n_periods'],
                 grouped_data['weights'])

        npt.assert_allclose(
            np.asarray(list(bbgb_params.values())).astype(float),
            np.asarray(bbtf._unload_params('alpha', 'beta', 'gamma', 'delta')).astype(float),
            atol=0.1, rtol=1e-2)



@pytest.mark.parametrize("T,r,alpha,a,b,observation_period_end,freq,size", [
                        (100, 0.24, 4.41, 0.79, 2.43, '2019-1-1', 'D', 500),
                        ([400, 200, 5, 103, 198, 401], 0.24, 4.41, 0.79, 2.43, '2019-1-1', 'D', 6),
                        (100, 0.24, 4.41, 0.79, 2.43, '2019-1-1', 'h', 500)
                        ])
def test_beta_geometric_nbd_model_transactional_data(T, r, alpha, a, b, observation_period_end, freq, size):
    np.random.seed(188898)
    transaction_data = beta_geometric_nbd_model_transactional_data(
        T=T,r=r,alpha=alpha,a=a,b=b, observation_period_end=observation_period_end, freq=freq, size=size
    )
    actual = summary_data_from_transaction_data(transactions=transaction_data,
                                                customer_id_col='customer_id', datetime_col='date',
                                                observation_period_end=observation_period_end,
                                                freq=freq)
    np.random.seed(188898)
    expected = beta_geometric_nbd_model(T=T,r=r,alpha=alpha,a=a,b=b,size=size)[['frequency', 'recency', 'T']]
    expected['recency'] = expected['recency'].apply(np.ceil)
    expected = expected.reset_index(drop=True)
    actual = actual.reset_index(drop=True)
    assert expected.equals(actual)
