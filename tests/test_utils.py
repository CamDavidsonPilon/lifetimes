
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

@pytest.fixture()
def large_transaction_level_data():
    d = [
            [1, '2015-01-01'],
            [1, '2015-02-06'],
            [2, '2015-01-01'],
            [3, '2015-01-01'],
            [3, '2015-01-02'],
            [3, '2015-01-05'],
            [4, '2015-01-16'],
            [4, '2015-02-02'],
            [4, '2015-02-05'],
            [5, '2015-01-16'],
            [5, '2015-01-17'],
            [5, '2015-01-18'],
            [6, '2015-02-02'],
    ]
    return pd.DataFrame(d, columns=['id', 'date'])

def test_summary_data_from_transaction_data_returns_correct_results(transaction_level_data):
    today = '2015-02-07'
    actual = utils.summary_data_from_transaction_data(transaction_level_data, 'id', 'date', observation_period_end=today)
    expected = pd.DataFrame([[1, 1., 1., 6.],
                             [2, 0., 0., 37.],
                             [3, 2., 33., 37.]], columns=['id', 'frequency', 'recency', 'cohort']).set_index('id')
    assert_frame_equal(actual, expected)

def test_summary_data_from_transaction_data_with_specific_datetime_format(transaction_level_data):
    transaction_level_data['date'] = transaction_level_data['date'].map(lambda x: x.replace('-',''))
    format = '%Y%m%d'
    today = '20150207'
    actual = utils.summary_data_from_transaction_data(transaction_level_data, 'id', 'date', observation_period_end=today, datetime_format=format)
    expected = pd.DataFrame([[1, 1., 1., 6.],
                             [2, 0., 0., 37.],
                             [3, 2., 33., 37.]], columns=['id', 'frequency', 'recency', 'cohort']).set_index('id')
    assert_frame_equal(actual, expected)


def test_summary_data_from_transaction_data_converts_no_repeat_purchases_to_have_zero_recency(transaction_level_data):
    today = '2015-02-07'
    actual = utils.summary_data_from_transaction_data(transaction_level_data, 'id', 'date', observation_period_end=today)
    assert actual.ix[2]['frequency'] == 0
    assert actual.ix[2]['recency'] == 0


def test_calibration_and_holdout_data(large_transaction_level_data):
    today = '2015-02-07'
    calibration_end = '2015-02-01'
    actual = utils.calibration_and_holdout_data(large_transaction_level_data, 'id', 'date', calibration_end, observation_period_end=today)
    assert actual.ix[1]['frequency_holdout'] == 1
    assert actual.ix[2]['frequency_holdout'] == 0

    with pytest.raises(KeyError):
        actual.ix[6] 

def test_columns_returned_from_calibration_and_holdout_data(large_transaction_level_data):
    today = '2015-02-07'
    calibration_end = '2015-02-01'
    actual = utils.calibration_and_holdout_data(large_transaction_level_data, 'id', 'date', calibration_end, observation_period_end=today)
    assert sorted(list(actual.columns)) == ['cohort_cal', 'cohort_holdout', 'frequency_cal', 'frequency_holdout', 'recency_cal']

