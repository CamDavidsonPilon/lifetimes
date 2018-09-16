"""Test lifetimes utils."""
import pytest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_almost_equal, assert_allclose

from lifetimes import utils, BetaGeoFitter, ParetoNBDFitter
from lifetimes.datasets import load_dataset


@pytest.fixture()
def example_transaction_data():
    return pd.read_csv('lifetimes/datasets/example_transactions.csv', parse_dates=['date'])


@pytest.fixture()
def example_summary_data(example_transaction_data):
    return utils.summary_data_from_transaction_data(example_transaction_data, 'id', 'date', observation_period_end=max(example_transaction_data.date))


@pytest.fixture()
def fitted_bg(example_summary_data):
    bg = BetaGeoFitter()
    bg.fit(example_summary_data['frequency'], example_summary_data['recency'], example_summary_data['T'], iterative_fitting=2, tol=1e-6)
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


@pytest.fixture()
def cdnow_transactions():
    transactions = load_dataset('CDNOW_sample.txt', header=None, sep=r'\s+')
    transactions.columns = ['id_total', 'id_sample', 'date', 'num_cd_purc',
                            'total_value']
    return transactions[['id_sample', 'date']]


@pytest.fixture()
def df_cum_transactions(cdnow_transactions):
    datetime_col = 'date'
    customer_id_col = 'id_sample'
    t = 25 * 7
    datetime_format = '%Y%m%d'
    freq = 'D'
    observation_period_end = '19970930'
    freq_multiplier = 7

    transactions_summary = utils.summary_data_from_transaction_data(
        cdnow_transactions, customer_id_col, datetime_col,
        datetime_format=datetime_format, freq=freq, freq_multiplier=freq_multiplier,
        observation_period_end=observation_period_end)

    transactions_summary = transactions_summary.reset_index()

    model = ParetoNBDFitter()
    model.fit(transactions_summary['frequency'],
              transactions_summary['recency'],
              transactions_summary['T'])

    df_cum = utils.expected_cumulative_transactions(
        model, cdnow_transactions, datetime_col, customer_id_col, t,
        datetime_format, freq, set_index_date=False, freq_multiplier=freq_multiplier)
    return df_cum


def test_find_first_transactions_returns_correct_results(large_transaction_level_data):
    today = '2015-02-07'
    actual = utils._find_first_transactions(large_transaction_level_data, 'id', 'date', observation_period_end=today)
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
                             [6, pd.Period('2015-02-02', 'D'), True]], columns=['id', 'date', 'first'])
    assert_frame_equal(actual, expected)


def test_find_first_transactions_with_specific_non_daily_frequency(large_transaction_level_data):
    today = '2015-02-07'
    actual = utils._find_first_transactions(large_transaction_level_data, 'id', 'date', observation_period_end=today, freq='W')
    expected = pd.DataFrame([[1, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), True],
                             [1, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), False],
                             [2, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), True],
                             [3, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), True],
                             [3, pd.Period('2015-01-05/2015-01-11', 'W-SUN'), False],
                             [4, pd.Period('2015-01-12/2015-01-18', 'W-SUN'), True],
                             [4, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), False],
                             [5, pd.Period('2015-01-12/2015-01-18', 'W-SUN'), True],
                             [6, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), True]],
                            columns=['id', 'date', 'first'],
                            index=actual.index)  # we shouldn't really care about row ordering or indexing, but assert_frame_equals is strict about it
    assert_frame_equal(actual, expected)


def test_find_first_transactions_with_monetary_values(large_transaction_level_data_with_monetary_value):
    today = '2015-02-07'
    actual = utils._find_first_transactions(large_transaction_level_data_with_monetary_value, 'id', 'date', 'monetary_value', observation_period_end=today)
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
                             [6, pd.Period('2015-02-02', 'D'), 5, True]], columns=['id', 'date', 'monetary_value', 'first'])
    assert_frame_equal(actual, expected)


def test_find_first_transactions_with_monetary_values_with_specific_non_daily_frequency(large_transaction_level_data_with_monetary_value):
    today = '2015-02-07'
    actual = utils._find_first_transactions(large_transaction_level_data_with_monetary_value, 'id', 'date', 'monetary_value', observation_period_end=today, freq='W')
    expected = pd.DataFrame([[1, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), 1, True],
                             [1, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), 2, False],
                             [2, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), 2, True],
                             [3, pd.Period('2014-12-29/2015-01-04', 'W-SUN'), 4, True],
                             [3, pd.Period('2015-01-05/2015-01-11', 'W-SUN'), 5, False],
                             [4, pd.Period('2015-01-12/2015-01-18', 'W-SUN'), 6, True],
                             [4, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), 6, False],
                             [5, pd.Period('2015-01-12/2015-01-18', 'W-SUN'), 12, True],
                             [6, pd.Period('2015-02-02/2015-02-08', 'W-SUN'), 5, True]], columns=['id', 'date', 'monetary_value', 'first'])
    assert_frame_equal(actual, expected)


def test_summary_data_from_transaction_data_returns_correct_results(transaction_level_data):
    today = '2015-02-07'
    actual = utils.summary_data_from_transaction_data(transaction_level_data, 'id', 'date', observation_period_end=today)
    expected = pd.DataFrame([[1, 1., 5., 6.],
                             [2, 0., 0., 37.],
                             [3, 2., 4., 37.]], columns=['id', 'frequency', 'recency', 'T']).set_index('id')
    assert_frame_equal(actual, expected)


def test_summary_data_from_transaction_data_works_with_string_customer_ids(transaction_level_data):
    d = [
        ['X', '2015-02-01'],
        ['X', '2015-02-06'],
        ['Y', '2015-01-01'],
        ['Y', '2015-01-01'],
        ['Y', '2015-01-02'],
        ['Y', '2015-01-05'],
    ]
    df = pd.DataFrame(d, columns=['id', 'date'])
    utils.summary_data_from_transaction_data(df, 'id', 'date')


def test_summary_data_from_transaction_data_works_with_int_customer_ids_and_doesnt_coerce_to_float(transaction_level_data):
    d = [
        [1, '2015-02-01'],
        [1, '2015-02-06'],
        [1, '2015-01-01'],
        [2, '2015-01-01'],
        [2, '2015-01-02'],
        [2, '2015-01-05'],
    ]
    df = pd.DataFrame(d, columns=['id', 'date'])
    actual = utils.summary_data_from_transaction_data(df, 'id', 'date')
    assert actual.index.dtype == 'int64'


def test_summary_data_from_transaction_data_with_specific_datetime_format(transaction_level_data):
    transaction_level_data['date'] = transaction_level_data['date'].map(lambda x: x.replace('-', ''))
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


def test_summary_data_from_transaction_data_will_choose_the_correct_first_order_to_drop_in_monetary_transactions():
    # this is the correct behaviour. See https://github.com/CamDavidsonPilon/lifetimes/issues/85
    # and test_summary_statistics_are_indentical_to_hardies_paper_confirming_correct_aggregations
    cust = pd.Series([2, 2, 2])
    dates_ordered = pd.to_datetime(pd.Series([
                  '2014-03-14 00:00:00',
                  '2014-04-09 00:00:00',
                  '2014-05-21 00:00:00']))
    sales = pd.Series([10, 20, 25])
    transaction_data = pd.DataFrame({'date': dates_ordered, 'id': cust, 'sales': sales})
    summary_ordered_data = utils.summary_data_from_transaction_data(transaction_data, 'id', 'date', 'sales')

    dates_unordered = pd.to_datetime(pd.Series([
                  '2014-04-09 00:00:00',
                  '2014-03-14 00:00:00',
                  '2014-05-21 00:00:00']))
    sales = pd.Series([20, 10, 25])
    transaction_data = pd.DataFrame({'date': dates_unordered, 'id': cust, 'sales': sales})
    summary_unordered_data = utils.summary_data_from_transaction_data(transaction_data, 'id', 'date', 'sales')

    assert_frame_equal(summary_ordered_data, summary_unordered_data)
    assert summary_ordered_data['monetary_value'].loc[2] == 22.5


def test_summary_statistics_are_indentical_to_hardies_paper_confirming_correct_aggregations():
    # see http://brucehardie.com/papers/rfm_clv_2005-02-16.pdf
    # RFM and CLV: Using Iso-value Curves for Customer Base Analysis
    df = pd.read_csv('lifetimes/datasets/CDNOW_sample.txt', sep='\s+', header=None, names=['_id', 'id', 'date', 'cds_bought', 'spent'])
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df_train = df[df['date'] < '1997-10-01']
    summary = utils.summary_data_from_transaction_data(df_train, 'id', 'date', 'spent')
    results = summary[summary['frequency'] > 0]['monetary_value'].describe()

    assert np.round(results.loc['mean']) == 35
    assert np.round(results.loc['std']) == 30
    assert np.round(results.loc['min']) == 3
    assert np.round(results.loc['50%']) == 27
    assert np.round(results.loc['max']) == 300
    assert np.round(results.loc['count']) == 946


def test_calibration_and_holdout_data(large_transaction_level_data):
    today = '2015-02-07'
    calibration_end = '2015-02-01'
    actual = utils.calibration_and_holdout_data(large_transaction_level_data, 'id', 'date', calibration_end, observation_period_end=today)
    assert actual.loc[1]['frequency_holdout'] == 1
    assert actual.loc[2]['frequency_holdout'] == 0

    with pytest.raises(KeyError):
        actual.loc[6]


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
        [1, '2015-02-06'],  # excluded from both holdout and calibration
        [2, '2015-01-01'],
        [3, '2015-01-01'],
        [3, '2015-01-02'],
        [3, '2015-01-05'],
        [4, '2015-01-16'],
        [4, '2015-02-02'],
        [4, '2015-02-05'],  # excluded from both holdout and calibration
        [5, '2015-01-16'],
        [5, '2015-01-17'],
        [5, '2015-01-18'],
        [6, '2015-02-02'],
    ]
    transactions = pd.DataFrame(d, columns=['id', 'date'])
    actual = utils.calibration_and_holdout_data(transactions, 'id', 'date', calibration_period_end='2015-02-01', observation_period_end='2015-02-04')
    assert actual['frequency_holdout'].loc[1] == 0
    assert actual['frequency_holdout'].loc[4] == 1


def test_calibration_and_holdout_data_with_monetary_value(large_transaction_level_data_with_monetary_value):
    today = '2015-02-07'
    calibration_end = '2015-02-01'
    actual = utils.calibration_and_holdout_data(large_transaction_level_data_with_monetary_value,
                                                'id',
                                                'date',
                                                calibration_end,
                                                observation_period_end=today,
                                                monetary_value_col='monetary_value')
    assert (actual['monetary_value_cal'] == [0, 0, 3, 0, 4.5]).all()
    assert (actual['monetary_value_holdout'] == [2, 0, 0, 3, 0]).all()


def test_summary_data_from_transaction_data_squashes_period_purchases_to_one_purchase():
    transactions = pd.DataFrame([[1, '2015-01-01'], [1, '2015-01-01']], columns=['id', 't'])
    actual = utils.summary_data_from_transaction_data(transactions, 'id', 't', freq='W')
    assert actual.loc[1]['frequency'] == 1. - 1.


def test_calculate_alive_path(example_transaction_data, example_summary_data, fitted_bg):
    user_data = example_transaction_data[example_transaction_data['id'] == 33]
    frequency, recency, T = example_summary_data.loc[33]
    alive_path = utils.calculate_alive_path(fitted_bg, user_data, 'date', 205)
    assert alive_path[0] == 1
    assert alive_path[T] == fitted_bg.conditional_probability_alive(frequency, recency, T)


def test_check_inputs():
    frequency = np.array([0, 1, 2])
    recency = np.array([0, 1, 10])
    T = np.array([5, 6, 15])
    monetary_value = np.array([2.3, 490, 33.33])
    assert utils._check_inputs(frequency, recency, T, monetary_value) is None

    with pytest.raises(ValueError):
        bad_recency = T + 1
        utils._check_inputs(frequency, bad_recency, T)

    with pytest.raises(ValueError):
        bad_recency = recency.copy()
        bad_recency[0] = 1
        utils._check_inputs(frequency, bad_recency, T)

    with pytest.raises(ValueError):
        bad_freq = np.array([0, 0.5, 2])
        utils._check_inputs(bad_freq, recency, T)

    with pytest.raises(ValueError):
        bad_monetary_value = monetary_value.copy()
        bad_monetary_value[0] = 0
        utils._check_inputs(frequency, recency, T, bad_monetary_value)


def test_summary_data_from_transaction_data_obeys_data_contraints(example_summary_data):
    assert utils._check_inputs(example_summary_data['frequency'], example_summary_data['recency'], example_summary_data['T']) is None


def test_scale_time():
    max_T = 200.
    T = np.arange(max_T)
    assert utils._scale_time(T) == 1. / (max_T - 1)


def test_customer_lifetime_value_with_known_values(fitted_bg):
    """
    >>> print fitted_bg
    <lifetimes.BetaGeoFitter: fitted with 5000 subjects, r: 0.16, alpha: 1.86, a: 1.85, b: 3.18>
    >>> t = fitted_bg.data.head()
    >>> t
       frequency  recency    T
       0          0        0  298
       1          0        0  224
       2          6      142  292
       3          0        0  147
       4          2        9  183
    >>> print fitted_bg.predict(30, t['frequency'], t['recency'], t['T'])
    0    0.016053
    1    0.021171
    2    0.030461
    3    0.031686
    4    0.001607
    dtype: float64
    """
    t = fitted_bg.data.head()
    expected = np.array([0.016053, 0.021171, 0.030461, 0.031686, 0.001607])
    # discount_rate=0 means the clv will be the same as the predicted
    clv_d0 = utils._customer_lifetime_value(fitted_bg, t['frequency'], t['recency'], t['T'], monetary_value=pd.Series([1, 1, 1, 1, 1]), time=1, discount_rate=0.)
    assert_almost_equal(clv_d0.values, expected, decimal=5)
    # discount_rate=1 means the clv will halve over a period
    clv_d1 = utils._customer_lifetime_value(fitted_bg, t['frequency'], t['recency'], t['T'], monetary_value=pd.Series([1, 1, 1, 1, 1]), time=1, discount_rate=1.)
    assert_almost_equal(clv_d1.values, expected / 2., decimal=5)
    # time=2, discount_rate=0 means the clv will be twice the initial
    clv_t2_d0 = utils._customer_lifetime_value(fitted_bg, t['frequency'], t['recency'], t['T'], monetary_value=pd.Series([1, 1, 1, 1, 1]), time=2, discount_rate=0)
    assert_allclose(clv_t2_d0.values, expected * 2., rtol=0.1)
    # time=2, discount_rate=1 means the clv will be twice the initial
    clv_t2_d1 = utils._customer_lifetime_value(fitted_bg, t['frequency'], t['recency'], t['T'], monetary_value=pd.Series([1, 1, 1, 1, 1]), time=2, discount_rate=1.)
    assert_allclose(clv_t2_d1.values, expected / 2. + expected / 4., rtol=0.1)


def test_expected_cumulative_transactions_dedups_inside_a_time_period(fitted_bg, example_transaction_data):
    by_week = utils.expected_cumulative_transactions(fitted_bg, example_transaction_data, 'date', 'id', 10, freq='W')
    by_day = utils.expected_cumulative_transactions(fitted_bg, example_transaction_data, 'date', 'id', 10, freq='D')
    assert (by_week['actual'] >= by_day['actual']).all()


def test_expected_cumulative_transactions_equals_r_btyd_walktrough(df_cum_transactions):
    """
    Validate expected cumulative transactions with BTYD walktrough

    https://cran.r-project.org/web/packages/BTYD/vignettes/BTYD-walkthrough.pdf

    cum.tracking[,20:25]
    # [,1] [,2] [,3] [,4] [,5] [,6]
    # actual 1359 1414 1484 1517 1573 1672
    # expected 1309 1385 1460 1533 1604 1674

    """
    actual_btyd = [1359, 1414, 1484, 1517, 1573, 1672]
    expected_btyd = [1309, 1385, 1460, 1533, 1604, 1674]

    actual = df_cum_transactions['actual'].iloc[19:25].values
    predicted = df_cum_transactions['predicted'].iloc[19:25].values.round()

    assert_allclose(actual, actual_btyd)
    assert_allclose(predicted, expected_btyd)


def test_incremental_transactions_equals_r_btyd_walktrough(df_cum_transactions):
    """
    Validate incremental transactions with BTYD walktrough

    https://cran.r-project.org/web/packages/BTYD/vignettes/BTYD-walkthrough.pdf

    inc.tracking[,20:25]
    # [,1] [,2] [,3] [,4] [,5] [,6]
    # actual 73.00 55.00 70.00 33.00 56.00 99.00
    # expected 78.31 76.42 74.65 72.98 71.41 69.93

    """
    # get incremental from cumulative transactions
    df_inc_transactions = df_cum_transactions.apply(lambda x: x - x.shift(1))

    actual_btyd = [73.00, 55.00, 70.00, 33.00, 56.00, 99.00]
    expected_btyd = [78.31, 76.42, 74.65, 72.98, 71.41, 69.93]

    actual = df_inc_transactions['actual'].iloc[19:25].values
    predicted = df_inc_transactions['predicted'].iloc[19:25].values.round(2)

    assert_allclose(actual, actual_btyd)
    assert_allclose(predicted, expected_btyd, atol=1e-2)


def test_expected_cumulative_transactions_date_index(cdnow_transactions):
    """
    Test set_index as date for cumulative transactions and bgf fitter.

    Get first 14 cdnow transactions dates and validate that date index,
    freq_multiplier = 1 working and compare with tested data for last 4 records.

    dates = ['1997-01-11', '1997-01-12', '1997-01-13', '1997-01-14']
    actual_trans = [11, 12, 15, 19]
    expected_trans = [10.67, 12.67, 14.87, 17.24]

    """
    datetime_col = 'date'
    customer_id_col = 'id_sample'
    t = 14
    datetime_format = '%Y%m%d'
    freq = 'D'
    observation_period_end = '19970930'
    freq_multiplier = 1

    transactions_summary = utils.summary_data_from_transaction_data(
        cdnow_transactions, customer_id_col, datetime_col,
        datetime_format=datetime_format, freq=freq,
        freq_multiplier=freq_multiplier,
        observation_period_end=observation_period_end)

    transactions_summary = transactions_summary.reset_index()

    model = BetaGeoFitter()
    model.fit(transactions_summary['frequency'],
              transactions_summary['recency'],
              transactions_summary['T'])

    df_cum = utils.expected_cumulative_transactions(
        model, cdnow_transactions, datetime_col, customer_id_col, t,
        datetime_format, freq, set_index_date=True,
        freq_multiplier=freq_multiplier)

    dates = ['1997-01-11', '1997-01-12', '1997-01-13', '1997-01-14']
    actual_trans = [11, 12, 15, 19]
    expected_trans = [10.67, 12.67, 14.87, 17.24]

    date_index = df_cum.iloc[-4:].index.to_timestamp().astype(str)
    actual = df_cum['actual'].iloc[-4:].values
    predicted = df_cum['predicted'].iloc[-4:].values.round(2)

    assert all(dates == date_index)
    assert_allclose(actual, actual_trans)
    assert_allclose(predicted, expected_trans, atol=1e-2)
