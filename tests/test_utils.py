
import pytest
import pandas as pd 
from pandas.util.testing import assert_frame_equal
from lifetimes import utils


@pytest.fixture()
def transaction_level_data():
    d = [
            [1, '2015-02-01'],
            [1, '2015-02-06'],
            [2, '2015-01-01'],
            [3, '2015-01-01'],
            [3, '2015-01-02'],
            [3, '2015-01-05'],
    ]
    return pd.DataFrame(d, columns=['id', 'date'])


def test_summary_data_from_transaction_data_returns_correct_results(transaction_level_data):
    today = '2015-02-07'
    actual = utils.summary_data_from_transaction_data(transaction_level_data, 'id', 'date', observation_period_end=today)
    expected = pd.DataFrame([[1, 1., 1., 6.],
                             [2, 0., 37., 37.],
                             [3, 2., 33., 37.]], columns=['id', 'x', 't_x', 'T']).set_index('id')
    assert_frame_equal(actual, expected)
