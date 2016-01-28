import pytest
import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal

from lifetimes import utils, BetaGeoFitter


@pytest.fixture()
def example_transaction_data():
    return pd.read_csv('lifetimes/datasets/example_transactions.csv', parse_dates=['date'])

@pytest.fixture()
def example_summary_data(example_transaction_data):
    return utils.summary_data_from_transaction_data(example_transaction_data, 'id', 'date', observation_period_end=max(example_transaction_data.date))

@pytest.fixture()
def fitted_bg(example_summary_data):
    bg = BetaGeoFitter()
    bg.fit(example_summary_data['frequency'], example_summary_data['recency'], example_summary_data['T'], iterative_fitting=0)
    return bg

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

@pytest.fixture()
def large_transaction_level_data_with_monetary_value():
    d = [
            [1, '2015-01-01', 1],
            [1, '2015-02-06', 2],
            [2, '2015-01-01', 2],
            [3, '2015-01-01', 3],
            [3, '2015-01-02', 1],
            [3, '2015-01-05', 5],
            [4, '2015-01-16', 6],
            [4, '2015-02-02', 3],
            [4, '2015-02-05', 3],
            [5, '2015-01-16', 3],
            [5, '2015-01-17', 1],
            [5, '2015-01-18', 8],
            [6, '2015-02-02', 5],
    ]
    return pd.DataFrame(d, columns=['id', 'date', 'monetary_value'])


def test_find_first_transactions_returns_correct_results(large_transaction_level_data):
    today = '2015-02-07'
    actual = utils.find_first_transactions(large_transaction_level_data, 'id', 'date', observation_period_end=today)
    expected = pd.DataFrame([[1, pd.Period('2015-01-01', 'D'), True],
                             [1, pd.Period('2015-02-06', 'D'), False],
                             [2, pd.Period('2015-01-01', 'D'), True],
                             [3, pd.Period('2015-01-01', 'D'), True],
                             [3, pd.Period('2015-01-02', 'D'), False],
                             [3, pd.Period('2015-01-05', 'D'), False],
                             [4, pd.Period('2015-01-16', 'D'), True],
                             [4, pd.Period('2015-02-02', 'D'), False],
                             [4, pd.Period('2015-02-05', 'D'), False],
                             [5, pd.Period('2015-01-16', 'D'), True],
                             [5, pd.Period('2015-01-17', 'D'), False],
                             [5, pd.Period('2015-01-18', 'D'), False],
                             [6, pd.Period('2015-02-02', 'D'), True]], columns=['id','date','first'])
    assert_frame_equal(actual, expected)


def test_find_first_transactions_with_specific_non_daily_frequency(large_transaction_level_data):
    today = '2015-02-07'
    actual = utils.find_first_transactions(large_transaction_level_data, 'id', 'date', observation_period_end=today, freq='W')
    expected = pd.DataFrame([[1, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), True],
                             [1, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), False],
                             [2, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), True],
                             [3, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), True],
                             [3, pd.Period('2015-01-05/2015-01-11', 'W-SUN'), False],
                             [4, pd.Period('2015-01-12/2015-01-18', 'W-SUN'), True],
                             [4, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), False],
                             [5, pd.Period('2015-01-12/2015-01-18', 'W-SUN'), True],
                             [6, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), True]],
                            columns=['id','date','first'],
                            index=actual.index) #we shouldn't really care about row ordering or indexing, but assert_frame_equals is strict about it
    assert_frame_equal(actual, expected)


def test_find_first_transactions_with_monetary_values(large_transaction_level_data_with_monetary_value):
    today = '2015-02-07'
    actual = utils.find_first_transactions(large_transaction_level_data_with_monetary_value, 'id', 'date', 'monetary_value', observation_period_end=today)
    expected = pd.DataFrame([[1, pd.Period('2015-01-01', 'D'), 1, True],
                             [1, pd.Period('2015-02-06', 'D'), 2, False],
                             [2, pd.Period('2015-01-01', 'D'), 2, True],
                             [3, pd.Period('2015-01-01', 'D'), 3, True],
                             [3, pd.Period('2015-01-02', 'D'), 1, False],
                             [3, pd.Period('2015-01-05', 'D'), 5, False],
                             [4, pd.Period('2015-01-16', 'D'), 6, True],
                             [4, pd.Period('2015-02-02', 'D'), 3, False],
                             [4, pd.Period('2015-02-05', 'D'), 3, False],
                             [5, pd.Period('2015-01-16', 'D'), 3, True],
                             [5, pd.Period('2015-01-17', 'D'), 1, False],
                             [5, pd.Period('2015-01-18', 'D'), 8, False],
                             [6, pd.Period('2015-02-02', 'D'), 5, True]], columns=['id','date','monetary_value','first'])
    assert_frame_equal(actual, expected)


def test_find_first_transactions_with_monetary_values_with_specific_non_daily_frequency(large_transaction_level_data_with_monetary_value):
    today = '2015-02-07'
    actual = utils.find_first_transactions(large_transaction_level_data_with_monetary_value, 'id', 'date', 'monetary_value', observation_period_end=today, freq='W')
    expected = pd.DataFrame([[1, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), 1, True],
                             [1, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), 2, False],
                             [2, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), 2, True],
                             [3, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), 4, True],
                             [3, pd.Period('2015-01-05/2015-01-11', 'W-SUN'), 5, False],
                             [4, pd.Period('2015-01-12/2015-01-18', 'W-SUN'), 6, True],
                             [4, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), 6, False],
                             [5, pd.Period('2015-01-12/2015-01-18', 'W-SUN'), 12, True],
                             [6, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), 5, True]], columns=['id','date','monetary_value','first'])
    assert_frame_equal(actual, expected)


def test_summary_data_from_transaction_data_returns_correct_results(transaction_level_data):
    today = '2015-02-07'
    actual = utils.summary_data_from_transaction_data(transaction_level_data, 'id', 'date', observation_period_end=today)
    expected = pd.DataFrame([[1, 1., 5., 6.],
                             [2, 0., 0., 37.],
                             [3, 2., 4., 37.]], columns=['id', 'frequency', 'recency', 'T']).set_index('id')
    assert_frame_equal(actual, expected)

def test_summary_data_from_transaction_data_with_specific_datetime_format(transaction_level_data):
    transaction_level_data['date'] = transaction_level_data['date'].map(lambda x: x.replace('-',''))
    format = '%Y%m%d'
    today = '20150207'
    actual = utils.summary_data_from_transaction_data(transaction_level_data, 'id', 'date', observation_period_end=today, datetime_format=format)
    expected = pd.DataFrame([[1, 1., 5., 6.],
                             [2, 0., 0., 37.],
                             [3, 2., 4., 37.]], columns=['id', 'frequency', 'recency', 'T']).set_index('id')
    assert_frame_equal(actual, expected)


def test_summary_date_from_transaction_data_with_specific_non_daily_frequency(large_transaction_level_data):
    today = '20150207'
    actual = utils.summary_data_from_transaction_data(large_transaction_level_data, 'id', 'date', observation_period_end=today, freq='W')
    expected = pd.DataFrame([[1, 1., 5., 5.],
                             [2, 0., 0., 5.],
                             [3, 1., 1., 5.],
                             [4, 1., 3., 3.],
                             [5, 0., 0., 3.],
                             [6, 0., 0., 0.]], columns=['id', 'frequency', 'recency', 'T']).set_index('id')
    assert_frame_equal(actual, expected)


def test_summary_date_from_transaction_with_monetary_values(large_transaction_level_data_with_monetary_value):
    today = '20150207'
    actual = utils.summary_data_from_transaction_data(large_transaction_level_data_with_monetary_value, 'id', 'date', monetary_value_col='monetary_value', observation_period_end=today)
    expected = pd.DataFrame([[1, 1., 36., 37., 2],
                             [2, 0.,  0., 37., 0],
                             [3, 2.,  4., 37., 3],
                             [4, 2., 20., 22., 3],
                             [5, 2.,  2., 22., 4.5],
                             [6, 0.,  0.,  5., 0]], columns=['id', 'frequency', 'recency', 'T', 'monetary_value']).set_index('id')
    assert_frame_equal(actual, expected)


def test_calibration_and_holdout_data(large_transaction_level_data):
    today = '2015-02-07'
    calibration_end = '2015-02-01'
    actual = utils.calibration_and_holdout_data(large_transaction_level_data, 'id', 'date', calibration_end, observation_period_end=today)
    assert actual.ix[1]['frequency_holdout'] == 1
    assert actual.ix[2]['frequency_holdout'] == 0

    with pytest.raises(KeyError):
        actual.ix[6] 

def test_calibration_and_holdout_data_works_with_specific_frequency(large_transaction_level_data):
    today = '2015-02-07'
    calibration_end = '2015-02-01'
    actual = utils.calibration_and_holdout_data(large_transaction_level_data, 'id', 'date', calibration_end, observation_period_end=today, freq='W')
    expected_cols = ['id', 'frequency_cal', 'recency_cal', 'T_cal', 'frequency_holdout', 'duration_holdout']
    expected = pd.DataFrame([[1, 0., 0., 4., 1, 1],
                             [2, 0., 0., 4., 0, 1],
                             [3, 1., 1., 4., 0, 1],
                             [4, 0., 0., 2., 1, 1],
                             [5, 0., 0., 2., 0, 1]], columns=expected_cols).set_index('id')
    assert_frame_equal(actual, expected, check_dtype=False)

def test_calibration_and_holdout_data_gives_correct_date_boundaries():

    d = [
            [1, '2015-01-01'],
            [1, '2015-02-06'], # excluded from both holdout and calibration
            [2, '2015-01-01'],
            [3, '2015-01-01'],
            [3, '2015-01-02'],
            [3, '2015-01-05'],
            [4, '2015-01-16'],
            [4, '2015-02-02'],
            [4, '2015-02-05'], # excluded from both holdout and calibration
            [5, '2015-01-16'],
            [5, '2015-01-17'],
            [5, '2015-01-18'],
            [6, '2015-02-02'],
    ]
    transactions = pd.DataFrame(d, columns=['id', 'date'])
    actual = utils.calibration_and_holdout_data(transactions, 'id', 'date', calibration_period_end='2015-02-01', observation_period_end='2015-02-04')
    assert actual['frequency_holdout'].ix[1] == 0 
    assert actual['frequency_holdout'].ix[4] == 1 


def test_summary_data_from_transaction_data_squashes_period_purchases_to_one_purchase():
    transactions = pd.DataFrame([[1, '2015-01-01'], [1, '2015-01-01']], columns=['id', 't'])
    actual = utils.summary_data_from_transaction_data(transactions, 'id', 't', freq='W')
    assert actual.ix[1]['frequency'] == 1. - 1.


def test_calculate_alive_path(example_transaction_data, example_summary_data, fitted_bg):
    user_data = example_transaction_data[example_transaction_data['id'] == 33]
    frequency, recency, T = example_summary_data.loc[33]
    alive_path = utils.calculate_alive_path(fitted_bg, user_data, 'date', 205)
    assert alive_path[0] == 1
    assert alive_path[T] == fitted_bg.conditional_probability_alive(frequency, recency, T)

def test_check_inputs():
    freq, recency, T = np.array([0,1,2]), np.array([0, 1, 10]), np.array([5, 6, 15])
    assert utils._check_inputs(freq, recency, T) is None

    with pytest.raises(ValueError):
        bad_recency = T + 1
        utils._check_inputs(freq, bad_recency, T)

    with pytest.raises(ValueError):
        bad_recency = recency.copy()
        bad_recency[0] = 1
        utils._check_inputs(freq, bad_recency, T)

    with pytest.raises(ValueError):
        bad_freq = np.array([0, 0.5, 2])
        utils._check_inputs(bad_freq, recency, T)


def test_summary_data_from_transaction_data_obeys_data_contraints(example_summary_data):
    assert utils._check_inputs(example_summary_data['frequency'], example_summary_data['recency'], example_summary_data['T']) is None

def test_scale_time():
    max_T = 200.
    T = np.arange(max_T)
    assert utils._scale_time(T) == 10. / (max_T-1)



