from __future__ import print_function

import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest
import os
from collections import OrderedDict


import lifetimes.estimation as estimation
import lifetimes.utils as utils
from lifetimes.datasets import load_cdnow_summary, load_cdnow_summary_data_with_monetary_value, load_donations,\
    load_transaction_data


@pytest.fixture
def cdnow_customers():
    return load_cdnow_summary()


PATH_SAVE_MODEL = './base_fitter.pkl'
PATH_SAVE_BGNBD_MODEL = './betageo_fitter.pkl'


class TestBaseFitter():
    def test_repr(self):
        base_fitter = estimation.BaseFitter()
        assert repr(base_fitter) == '<lifetimes.BaseFitter>'
        base_fitter.params_ = dict(x=12.3, y=42)
        base_fitter.data = np.array([1, 2, 3])
        assert repr(base_fitter) == '<lifetimes.BaseFitter: fitted with 3 subjects, x: 12.30, y: 42.00>'
        base_fitter.data = None
        assert repr(base_fitter) == '<lifetimes.BaseFitter: x: 12.30, y: 42.00>'

    def test_unload_params(self):
        base_fitter = estimation.BaseFitter()
        with pytest.raises(ValueError):
            base_fitter._unload_params()
        base_fitter.params_ = dict(x=12.3, y=42)
        npt.assert_array_almost_equal([12.3, 42], base_fitter._unload_params('x', 'y'))

    def test_save_load_model(self):
        base_fitter = estimation.BaseFitter()
        base_fitter.save_model(PATH_SAVE_MODEL)
        assert os.path.exists(PATH_SAVE_MODEL) == True

        base_fitter_saved = estimation.BaseFitter()
        base_fitter_saved.load_model(PATH_SAVE_MODEL)

        assert repr(base_fitter) == repr(base_fitter_saved)
        os.remove(PATH_SAVE_MODEL)


class TestBetaGeoBetaBinomFitter():

    @pytest.fixture()
    def donations(self):
        return load_donations()

    def test_params_out_is_close_to_Hardie_paper(self, donations):
        donations = donations
        bbtf = estimation.BetaGeoBetaBinomFitter()
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['periods'],
            donations['weights'],
        )
        expected = np.array([1.204, 0.750, 0.657, 2.783])
        npt.assert_array_almost_equal(expected, np.array(bbtf._unload_params('alpha','beta','gamma','delta')),
                                      decimal=2)


    def test_prob_alive_is_close_to_Hardie_paper_table_6(self, donations):
        """Table 6: P(Alive in 2002) as a Function of Recency and Frequency"""

        bbtf = estimation.BetaGeoBetaBinomFitter()
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['periods'],
            donations['weights'],
        )

        bbtf.data['prob_alive'] = bbtf.conditional_probability_alive(1, donations['frequency'], donations['recency'], donations['periods'])

        # Expected probabilities for last year 1995-0 repeat, 1999-2 repeat, 2001-6 repeat
        expected = np.array([0.11, 0.59, 0.93])
        prob_list = np.zeros(3)
        prob_list[0] = (bbtf.data[(bbtf.data['frequency'] == 0) & (bbtf.data['recency'] == 0)]['prob_alive'])
        prob_list[1] = (bbtf.data[(bbtf.data['frequency'] == 2) & (bbtf.data['recency'] == 4)]['prob_alive'])
        prob_list[2] = (bbtf.data[(bbtf.data['frequency'] == 6) & (bbtf.data['recency'] == 6)]['prob_alive'])
        npt.assert_array_almost_equal(expected, prob_list, decimal=2)

    def test_conditional_expectation_returns_same_value_as_Hardie_excel_sheet(self, donations):
        """
        Total from Hardie's Conditional Expectations (II) sheet.
        http://brucehardie.com/notes/010/BGBB_2011-01-20_XLSX.zip
        """

        bbtf = estimation.BetaGeoBetaBinomFitter()
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['periods'],
            donations['weights'],
        )
        pred_purchases = bbtf.conditional_expected_number_of_purchases_up_to_time(5, donations['frequency'], donations['recency'], donations['periods']) * donations['weights']
        expected = 12884.2 # Sum of column F Exp Tot
        npt.assert_almost_equal(expected, pred_purchases.sum(), decimal=0)

    def test_expected_purchases_in_n_periods_returns_same_value_as_Hardie_excel_sheet(self, donations):
        """Total expected from Hardie's In-Sample Fit sheet."""

        bbtf = estimation.BetaGeoBetaBinomFitter()
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['periods'],
            donations['weights'],
        )
        expected = np.array([3454.9, 1253.1]) # Cells C18 and C24
        estimated = bbtf.expected_number_of_transactions_in_first_n_periods(6).loc[[0,6]].values.flatten()
        npt.assert_almost_equal(expected, estimated, decimal=0)

    def test_fit_with_index(self, donations):

        bbtf = estimation.BetaGeoBetaBinomFitter()
        index = range(len(donations), 0, -1)
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['periods'],
            donations['weights'],
            index=index
        )
        assert (bbtf.data.index == index).all() == True

        bbtf = estimation.BetaGeoBetaBinomFitter()
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['periods'],
            donations['weights'],
            index=None
        )
        assert (bbtf.data.index == index).all() == False


    def test_fit_with_and_without_weights(self, donations):

        exploded_dataset = pd.DataFrame(columns=['frequency', 'recency', 'periods'])

        for _, row in donations.iterrows():
            exploded_dataset = exploded_dataset.append(
                pd.DataFrame(
                        [[row['frequency'], row['recency'], row['periods']]] * row['weights'],
                        columns = ['frequency', 'recency', 'periods']
                ))

        exploded_dataset = exploded_dataset.astype(np.int64)
        assert exploded_dataset.shape[0] == donations['weights'].sum()

        bbtf_noweights = estimation.BetaGeoBetaBinomFitter()
        bbtf_noweights.fit(
            exploded_dataset['frequency'],
            exploded_dataset['recency'],
            exploded_dataset['periods'],
        )

        bbtf = estimation.BetaGeoBetaBinomFitter()
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['periods'],
            donations['weights'],
        )

        npt.assert_array_almost_equal(
            np.array(bbtf_noweights._unload_params('alpha','beta','gamma','delta')),
            np.array(bbtf._unload_params('alpha','beta','gamma','delta')),
        decimal=4
        )


class TestGammaGammaFitter():

    @pytest.fixture()
    def cdnow_customers_with_monetary_value(self):
        return load_cdnow_summary_data_with_monetary_value()

    def test_params_out_is_close_to_Hardie_paper(self, cdnow_customers_with_monetary_value):
        returning_cdnow_customers_with_monetary_value = cdnow_customers_with_monetary_value[
            cdnow_customers_with_monetary_value['frequency'] > 0
        ]
        ggf = estimation.GammaGammaFitter()
        ggf.fit(
            returning_cdnow_customers_with_monetary_value['frequency'],
            returning_cdnow_customers_with_monetary_value['monetary_value'],
            iterative_fitting=3
        )
        expected = np.array([6.25, 3.74, 15.44])
        npt.assert_array_almost_equal(expected, np.array(ggf._unload_params('p', 'q', 'v')), decimal=2)

    def test_conditional_expected_average_profit(self, cdnow_customers_with_monetary_value):

        ggf = estimation.GammaGammaFitter()
        ggf.params_ = OrderedDict({'p':6.25, 'q':3.74, 'v':15.44})

        summary = cdnow_customers_with_monetary_value.head(10)
        estimates = ggf.conditional_expected_average_profit(summary['frequency'], summary['monetary_value'])
        expected = np.array([24.65, 18.91, 35.17, 35.17, 35.17, 71.46, 18.91, 35.17, 27.28, 35.17]) # from Hardie spreadsheet http://brucehardie.com/notes/025/

        npt.assert_allclose(estimates.values, expected, atol=0.1)

    def test_customer_lifetime_value_with_bgf(self, cdnow_customers_with_monetary_value):

        ggf = estimation.GammaGammaFitter()
        ggf.params_ = OrderedDict({'p':6.25, 'q':3.74, 'v':15.44})

        bgf = estimation.BetaGeoFitter()
        bgf.fit(cdnow_customers_with_monetary_value['frequency'],
                cdnow_customers_with_monetary_value['recency'],
                cdnow_customers_with_monetary_value['T'],
                iterative_fitting=3)

        ggf_clv = ggf.customer_lifetime_value(
                bgf,
                cdnow_customers_with_monetary_value['frequency'],
                cdnow_customers_with_monetary_value['recency'],
                cdnow_customers_with_monetary_value['T'],
                cdnow_customers_with_monetary_value['monetary_value']
        )

        utils_clv = utils._customer_lifetime_value(
                bgf,
                cdnow_customers_with_monetary_value['frequency'],
                cdnow_customers_with_monetary_value['recency'],
                cdnow_customers_with_monetary_value['T'],
                ggf.conditional_expected_average_profit(cdnow_customers_with_monetary_value['frequency'],
                                                        cdnow_customers_with_monetary_value['monetary_value'])
        )
        npt.assert_equal(ggf_clv.values, utils_clv.values)

        ggf_clv = ggf.customer_lifetime_value(
            bgf,
            cdnow_customers_with_monetary_value["frequency"],
            cdnow_customers_with_monetary_value["recency"],
            cdnow_customers_with_monetary_value["T"],
            cdnow_customers_with_monetary_value["monetary_value"],
            freq="H",
        )

        utils_clv = utils._customer_lifetime_value(
            bgf,
            cdnow_customers_with_monetary_value["frequency"],
            cdnow_customers_with_monetary_value["recency"],
            cdnow_customers_with_monetary_value["T"],
            ggf.conditional_expected_average_profit(
                cdnow_customers_with_monetary_value["frequency"], cdnow_customers_with_monetary_value["monetary_value"]
            ),
            freq="H",
        )
        npt.assert_equal(ggf_clv.values, utils_clv.values)

    def test_fit_with_index(self, cdnow_customers_with_monetary_value):
        returning_cdnow_customers_with_monetary_value = cdnow_customers_with_monetary_value[
            cdnow_customers_with_monetary_value['frequency'] > 0
        ]

        ggf = estimation.GammaGammaFitter()
        index = range(len(returning_cdnow_customers_with_monetary_value), 0, -1)
        ggf.fit(
            returning_cdnow_customers_with_monetary_value['frequency'],
            returning_cdnow_customers_with_monetary_value['monetary_value'],
            iterative_fitting=1,
            index=index
        )
        assert (ggf.data.index == index).all() == True

        ggf = estimation.GammaGammaFitter()
        ggf.fit(
            returning_cdnow_customers_with_monetary_value['frequency'],
            returning_cdnow_customers_with_monetary_value['monetary_value'],
            iterative_fitting=1,
            index=None
        )
        assert (ggf.data.index == index).all() == False

    def test_params_out_is_close_to_Hardie_paper_with_q_constraint(self, cdnow_customers_with_monetary_value):
        returning_cdnow_customers_with_monetary_value = cdnow_customers_with_monetary_value[
            cdnow_customers_with_monetary_value['frequency'] > 0
        ]
        ggf = estimation.GammaGammaFitter()
        ggf.fit(
            returning_cdnow_customers_with_monetary_value['frequency'],
            returning_cdnow_customers_with_monetary_value['monetary_value'],
            iterative_fitting=3,
            q_constraint=True
        )
        expected = np.array([6.25, 3.74, 15.44])
        npt.assert_array_almost_equal(expected, np.array(ggf._unload_params('p', 'q', 'v')), decimal=2)

    def test_negative_log_likelihood_is_inf_when_q_constraint_true_and_q_lt_one(self):
        frequency = 25
        avg_monetary_value = 100
        ggf = estimation.GammaGammaFitter()
        assert np.isinf(ggf._negative_log_likelihood([6.25, -3.75, 15.44], frequency, avg_monetary_value, q_constraint=True))



class TestParetoNBDFitter():

    def test_overflow_error(self):

        ptf = estimation.ParetoNBDFitter()
        params = np.array([10.465, 7.98565181e-03, 3.0516, 2.820])
        freq = np.array([400., 500., 500.])
        rec = np.array([5., 1., 4.])
        age = np.array([6., 37., 37.])
        assert all([r < 0 and not np.isinf(r) and not pd.isnull(r)
                    for r in ptf._log_A_0(params, freq, rec, age)])

    def test_sum_of_scalar_inputs_to_negative_log_likelihood_is_equal_to_array(self):
        ptf = estimation.ParetoNBDFitter
        x = np.array([1, 3])
        t_x = np.array([2, 2])
        weights = np.array([1., 1.])
        t = np.array([5, 6])
        params = [1, 1, 1, 1]
        assert ptf()._negative_log_likelihood(params, np.array([x[0]]), np.array([t_x[0]]), np.array([t[0]]), weights[0], 0) \
            + ptf()._negative_log_likelihood(params, np.array([x[1]]), np.array([t_x[1]]), np.array([t[1]]), weights[0], 0) \
            == 2 * ptf()._negative_log_likelihood(params, x, t_x, t, weights, 0)

    def test_params_out_is_close_to_Hardie_paper(self, cdnow_customers):
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=3)
        expected = np.array([ 0.553, 10.578, 0.606, 11.669])
        npt.assert_array_almost_equal(expected, np.array(ptf._unload_params('r', 'alpha', 's', 'beta')), decimal=2)

    def test_expectation_returns_same_value_as_R_BTYD(self, cdnow_customers):
        """ From https://cran.r-project.org/web/packages/BTYD/BTYD.pdf """
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], tol=1e-6)

        expected = np.array([0.00000000, 0.05077821, 0.09916088, 0.14542507, 0.18979930,
            0.23247466, 0.27361274, 0.31335159, 0.35181024, 0.38909211])
        actual = ptf.expected_number_of_purchases_up_to_time(range(10))
        npt.assert_allclose(expected, actual, atol=0.01)

    def test_conditional_expectation_returns_same_value_as_R_BTYD(self, cdnow_customers):
        """ From https://cran.r-project.org/web/packages/BTYD/vignettes/BTYD-walkthrough.pdf """
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        x = 26.00
        t_x = 30.86
        T = 31
        t = 52
        expected =  25.46
        actual = ptf.conditional_expected_number_of_purchases_up_to_time(t, x, t_x, T)
        assert abs(expected - actual) < 0.01

    def test_conditional_expectation_underflow(self):
        """ Test a pair of inputs for the ParetoNBD ptf.conditional_expected_number_of_purchases_up_to_time().
            For a small change in the input, the result shouldn't change dramatically -- however, if the
            function doesn't guard against numeric underflow, this change in input will result in an
            underflow error.
        """
        ptf = estimation.ParetoNBDFitter()
        alpha = 10.58
        beta = 11.67
        r = 0.55
        s = 0.61
        ptf.params_ = OrderedDict({'alpha':alpha, 'beta':beta, 'r':r, 's':s})

        # small change in inputs
        left = ptf.conditional_expected_number_of_purchases_up_to_time(10, 132, 200, 200) # 6.2060517889632418
        right = ptf.conditional_expected_number_of_purchases_up_to_time(10, 133, 200, 200) # 6.2528722475748113
        assert abs(left - right) < 0.05

    def test_conditional_probability_alive_is_between_0_and_1(self, cdnow_customers):
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        for freq in np.arange(0, 100, 10.):
            for recency in np.arange(0, 100, 10.):
                for t in np.arange(recency, 100, 10.):
                    assert 0.0 <= ptf.conditional_probability_alive(freq, recency, t) <= 1.0

    def test_conditional_probability_alive(self, cdnow_customers):
        """
        Target taken from page 8,
        https://cran.r-project.org/web/packages/BTYD/vignettes/BTYD-walkthrough.pdf
        """
        ptf = estimation.ParetoNBDFitter()
        ptf.params_ = OrderedDict(
            zip(['r', 'alpha', 's', 'beta'],
                [0.5534, 10.5802, 0.6061, 11.6562]))
        p_alive = ptf.conditional_probability_alive(26.00, 30.86, 31.00)
        assert abs(p_alive - 0.9979) < 0.001

    def test_conditional_probability_alive_overflow_error(self):
        ptf = estimation.ParetoNBDFitter()
        ptf.params_ = OrderedDict(
            zip(['r', 'alpha', 's', 'beta'],
                [10.465, 7.98565181e-03, 3.0516, 2.820]))
        freq = np.array([40., 50., 50.])
        rec = np.array([5., 1., 4.])
        age = np.array([6., 37., 37.])
        assert all([r <= 1 and r >= 0 and not np.isinf(r) and not pd.isnull(r)
                    for r in ptf.conditional_probability_alive(freq, rec, age)])

    def test_conditional_probability_alive_matrix(self, cdnow_customers):
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        Z = ptf.conditional_probability_alive_matrix()
        max_t = int(ptf.data['T'].max())

        for t_x in range(Z.shape[0]):
            for x in range(Z.shape[1]):
                assert Z[t_x][x] == ptf.conditional_probability_alive(x, t_x, max_t)

    def test_fit_with_index(self, cdnow_customers):
        ptf = estimation.ParetoNBDFitter()
        index = range(len(cdnow_customers), 0, -1)
        ptf.fit(
            cdnow_customers['frequency'],
            cdnow_customers['recency'],
            cdnow_customers['T'],
            index=index
        )
        assert (ptf.data.index == index).all() == True

        ptf = estimation.ParetoNBDFitter()
        ptf.fit(
            cdnow_customers['frequency'],
            cdnow_customers['recency'],
            cdnow_customers['T'],
            index=None
        )
        assert (ptf.data.index == index).all() == False

    def test_conditional_probability_of_n_purchases_up_to_time_is_between_0_and_1(self, cdnow_customers):
        """
        Due to the large parameter space we take a random subset.
        """
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        for freq in np.random.choice(100, 5):
            for recency in np.random.choice(100, 5):
                for age in recency + np.random.choice(100, 5):
                    for t in np.random.choice(100, 5):
                        for n in np.random.choice(10, 5):
                            assert (
                                0.0
                                <= ptf.conditional_probability_of_n_purchases_up_to_time(n, t, freq, recency, age)
                                <= 1.0
                            )

    def test_conditional_probability_of_n_purchases_up_to_time_adds_up_to_1(self, cdnow_customers):
        """
        Due to the large parameter space we take a random subset. We also restrict our limits to keep the number of
        values of n for which the probability needs to be calculated to a sane level.
        """
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        for freq in np.random.choice(10, 5):
            for recency in np.random.choice(9, 5):
                for age in np.random.choice(np.arange(recency, 10, 1), 5):
                    for t in 1 + np.random.choice(9, 5):
                        npt.assert_almost_equal(
                            np.sum([
                                ptf.conditional_probability_of_n_purchases_up_to_time(n, t, freq, recency, age)
                                for n in np.arange(0, 20, 1)
                            ]),
                            1.0,
                            decimal=2
                        )


    def test_fit_with_and_without_weights(self, cdnow_customers):
        original_dataset_with_weights = cdnow_customers.copy()
        original_dataset_with_weights = original_dataset_with_weights.groupby(['frequency', 'recency', 'T']).size()
        original_dataset_with_weights = original_dataset_with_weights.reset_index()
        original_dataset_with_weights = original_dataset_with_weights.rename(columns={0:'weights'})

        pnbd_noweights = estimation.ParetoNBDFitter()
        pnbd_noweights.fit(
            cdnow_customers['frequency'],
            cdnow_customers['recency'],
            cdnow_customers['T'],
        )

        pnbd = estimation.ParetoNBDFitter()
        pnbd.fit(
            original_dataset_with_weights['frequency'],
            original_dataset_with_weights['recency'],
            original_dataset_with_weights['T'],
            original_dataset_with_weights['weights'],
        )

        npt.assert_array_almost_equal(
            np.array(pnbd_noweights._unload_params('r', 'alpha', 's', 'beta')),
            np.array(pnbd._unload_params('r', 'alpha', 's', 'beta')),
        decimal=2
        )

class TestBetaGeoFitter():

    def test_sum_of_scalar_inputs_to_negative_log_likelihood_is_equal_to_array(self):
        bgf = estimation.BetaGeoFitter
        x = np.array([1, 3])
        t_x = np.array([2, 2])
        t = np.array([5, 6])
        weights = np.array([1])
        params = [1, 1, 1, 1]
        assert bgf._negative_log_likelihood(params, x[0], np.array([t_x[0]]), np.array([t[0]]), weights[0], 0) \
            + bgf._negative_log_likelihood(params, x[1], np.array([t_x[1]]), np.array([t[1]]), weights[0], 0) \
            == 2 * bgf._negative_log_likelihood(params, x, t_x, t, weights, 0)

    def test_params_out_is_close_to_Hardie_paper(self, cdnow_customers):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=3)
        expected = np.array([0.243, 4.414, 0.793, 2.426])
        npt.assert_array_almost_equal(expected, np.array(bfg._unload_params('r', 'alpha', 'a', 'b')), decimal=3)

    def test_conditional_expectation_returns_same_value_as_Hardie_excel_sheet(self, cdnow_customers):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        x = 2
        t_x = 30.43
        T = 38.86
        t = 39
        expected = 1.226
        actual = bfg.conditional_expected_number_of_purchases_up_to_time(t, x, t_x, T)
        assert abs(expected - actual) < 0.001

    def test_expectation_returns_same_value_Hardie_excel_sheet(self, cdnow_customers):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], tol=1e-6)

        times = np.array([0.1429, 1.0, 3.00, 31.8571, 32.00, 78.00])
        expected = np.array([0.0078 ,0.0532 ,0.1506 ,1.0405,1.0437, 1.8576])
        actual = bfg.expected_number_of_purchases_up_to_time(times)
        npt.assert_array_almost_equal(actual, expected, decimal=3)

    def test_conditional_probability_alive_returns_1_if_no_repeat_purchases(self, cdnow_customers):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        assert bfg.conditional_probability_alive(0, 1, 1) == 1.0


    def test_conditional_probability_alive_is_between_0_and_1(self, cdnow_customers):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                for k in range(j, 100, 10):
                    assert 0 <= bfg.conditional_probability_alive(i, j, k) <= 1.0


    def test_fit_method_allows_for_better_accuracy_by_using_iterative_fitting(self, cdnow_customers):
        bfg1 = estimation.BetaGeoFitter()
        bfg2 = estimation.BetaGeoFitter()

        np.random.seed(0)
        bfg1.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        np.random.seed(0)
        bfg2.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=3)
        assert bfg1._negative_log_likelihood_ >= bfg2._negative_log_likelihood_


    def test_penalizer_term_will_shrink_coefs_to_0(self, cdnow_customers):
        bfg_no_penalizer = estimation.BetaGeoFitter()
        bfg_no_penalizer.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        params_1 = np.array(list(bfg_no_penalizer.params_.values()))

        bfg_with_penalizer = estimation.BetaGeoFitter(penalizer_coef=0.1)
        bfg_with_penalizer.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        params_2 = np.array(list(bfg_with_penalizer.params_.values()))
        assert np.all(params_2 < params_1)

        bfg_with_more_penalizer = estimation.BetaGeoFitter(penalizer_coef=10)
        bfg_with_more_penalizer.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        params_3 = np.array(list(bfg_with_more_penalizer.params_.values()))
        assert np.all(params_3 < params_2)


    def test_conditional_probability_alive_matrix(self, cdnow_customers):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        Z = bfg.conditional_probability_alive_matrix()
        max_t = int(bfg.data['T'].max())
        assert Z[0][0] == 1

        for t_x in range(Z.shape[0]):
            for x in range(Z.shape[1]):
                assert Z[t_x][x] == bfg.conditional_probability_alive(x, t_x, max_t)


    def test_probability_of_n_purchases_up_to_time_same_as_R_BTYD(self):
        """ See https://cran.r-project.org/web/packages/BTYD/BTYD.pdf """
        bgf = estimation.BetaGeoFitter()
        bgf.params_ = OrderedDict({'r':0.243, 'alpha':4.414, 'a':0.793, 'b':2.426})
        # probability that a customer will make 10 repeat transactions in the
        # time interval (0,2]
        expected = 1.07869e-07
        actual = bgf.probability_of_n_purchases_up_to_time(2,10)
        assert abs(expected - actual) < 10e-5
        # probability that a customer will make no repeat transactions in the
        # time interval (0,39]
        expected = 0.5737864
        actual = bgf.probability_of_n_purchases_up_to_time(39,0)
        assert abs(expected - actual) < 10e-5
        # PMF
        expected = np.array([0.0019995214, 0.0015170236, 0.0011633150, 0.0009003148, 0.0007023638,
                             0.0005517902, 0.0004361913, 0.0003467171, 0.0002769613, 0.0002222260])
        actual = np.array([bgf.probability_of_n_purchases_up_to_time(30,n) for n in range(11,21)])
        npt.assert_array_almost_equal(expected, actual, decimal=5)

    def test_scaling_inputs_gives_same_or_similar_results(self, cdnow_customers):
        bgf = estimation.BetaGeoFitter()
        bgf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        scale = 10
        bgf_with_large_inputs = estimation.BetaGeoFitter()
        bgf_with_large_inputs.fit(cdnow_customers['frequency'], scale * cdnow_customers['recency'], scale * cdnow_customers['T'], iterative_fitting=2)
        assert bgf_with_large_inputs._scale < 1.

        assert abs(bgf_with_large_inputs.conditional_probability_alive(1, scale * 1, scale * 2) - bgf.conditional_probability_alive(1, 1, 2)) < 10e-5
        assert abs(bgf_with_large_inputs.conditional_probability_alive(1, scale * 2, scale * 10) - bgf.conditional_probability_alive(1, 2, 10)) < 10e-5

    def test_save_load_bgnbd(self, cdnow_customers):
        """Test saving and loading model for BG/NBD."""
        bgf = estimation.BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        bgf.save_model(PATH_SAVE_BGNBD_MODEL)

        bgf_new = estimation.BetaGeoFitter()
        bgf_new.load_model(PATH_SAVE_BGNBD_MODEL)
        assert bgf_new.__dict__['penalizer_coef'] == bgf.__dict__['penalizer_coef']
        assert bgf_new.__dict__['_scale'] == bgf.__dict__['_scale']
        assert bgf_new.__dict__['params_'] == bgf.__dict__['params_']
        assert bgf_new.__dict__['_negative_log_likelihood_'] == bgf.__dict__['_negative_log_likelihood_']
        assert (bgf_new.__dict__['data'] == bgf.__dict__['data']).all().all()
        assert bgf_new.__dict__['predict'](1, 1, 2, 5) == bgf.__dict__['predict'](1, 1, 2, 5)
        assert bgf_new.expected_number_of_purchases_up_to_time(1) == bgf.expected_number_of_purchases_up_to_time(1)
        # remove saved model
        os.remove(PATH_SAVE_BGNBD_MODEL)

    def test_save_load_bgnbd_no_data(self, cdnow_customers):
        """Test saving and loading model for BG/NBD without data."""
        bgf = estimation.BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        bgf.save_model(PATH_SAVE_BGNBD_MODEL, save_data=False)

        bgf_new = estimation.BetaGeoFitter()
        bgf_new.load_model(PATH_SAVE_BGNBD_MODEL)
        assert bgf_new.__dict__['penalizer_coef'] == bgf.__dict__['penalizer_coef']
        assert bgf_new.__dict__['_scale'] == bgf.__dict__['_scale']
        assert bgf_new.__dict__['params_'] == bgf.__dict__['params_']
        assert bgf_new.__dict__['_negative_log_likelihood_'] == bgf.__dict__['_negative_log_likelihood_']
        assert bgf_new.__dict__['predict'](1, 1, 2, 5) == bgf.__dict__['predict'](1, 1, 2, 5)
        assert bgf_new.expected_number_of_purchases_up_to_time(1) == bgf.expected_number_of_purchases_up_to_time(1)

        assert bgf_new.__dict__['data'] is None
        # remove saved model
        os.remove(PATH_SAVE_BGNBD_MODEL)

    def test_save_load_bgnbd_no_generate_data(self, cdnow_customers):
        """Test saving and loading model for BG/NBD without generate_new_data method."""
        bgf = estimation.BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        bgf.save_model(PATH_SAVE_BGNBD_MODEL, save_generate_data_method=False)

        bgf_new = estimation.BetaGeoFitter()
        bgf_new.load_model(PATH_SAVE_BGNBD_MODEL)
        assert bgf_new.__dict__['penalizer_coef'] == bgf.__dict__['penalizer_coef']
        assert bgf_new.__dict__['_scale'] == bgf.__dict__['_scale']
        assert bgf_new.__dict__['params_'] == bgf.__dict__['params_']
        assert bgf_new.__dict__['_negative_log_likelihood_'] == bgf.__dict__['_negative_log_likelihood_']
        assert bgf_new.__dict__['predict'](1, 1, 2, 5) == bgf.__dict__['predict'](1, 1, 2, 5)
        assert bgf_new.expected_number_of_purchases_up_to_time(1) == bgf.expected_number_of_purchases_up_to_time(1)

        assert bgf_new.__dict__['generate_new_data'] is None
        # remove saved model
        os.remove(PATH_SAVE_BGNBD_MODEL)

    def test_save_load_bgnbd_no_data_replace_with_empty_str(self, cdnow_customers):
        """Test saving and loading model for BG/NBD without data with replaced value empty str."""
        bgf = estimation.BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        bgf.save_model(PATH_SAVE_BGNBD_MODEL, save_data=False, values_to_save=[''])

        bgf_new = estimation.BetaGeoFitter()
        bgf_new.load_model(PATH_SAVE_BGNBD_MODEL)
        assert bgf_new.__dict__['penalizer_coef'] == bgf.__dict__['penalizer_coef']
        assert bgf_new.__dict__['_scale'] == bgf.__dict__['_scale']
        assert bgf_new.__dict__['params_'] == bgf.__dict__['params_']
        assert bgf_new.__dict__['_negative_log_likelihood_'] == bgf.__dict__['_negative_log_likelihood_']
        assert bgf_new.__dict__['predict'](1, 1, 2, 5) == bgf.__dict__['predict'](1, 1, 2, 5)
        assert bgf_new.expected_number_of_purchases_up_to_time(1) == bgf.expected_number_of_purchases_up_to_time(1)

        assert bgf_new.__dict__['data'] is ''
        # remove saved model
        os.remove(PATH_SAVE_BGNBD_MODEL)

    def test_fit_with_index(self, cdnow_customers):
        bgf = estimation.BetaGeoFitter(penalizer_coef=0.0)
        index = range(len(cdnow_customers), 0, -1)
        bgf.fit(
            cdnow_customers['frequency'],
            cdnow_customers['recency'],
            cdnow_customers['T'],
            index=index
        )
        assert (bgf.data.index == index).all() == True

        bgf = estimation.BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(
            cdnow_customers['frequency'],
            cdnow_customers['recency'],
            cdnow_customers['T'],
            index=None
        )
        assert (bgf.data.index == index).all() == False

    def test_no_runtime_warnings_high_frequency(self, cdnow_customers):
        old_settings = np.seterr(all='raise')
        bgf = estimation.BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(
            cdnow_customers['frequency'],
            cdnow_customers['recency'],
            cdnow_customers['T'],
            index=None
        )

        p_alive = bgf.conditional_probability_alive(frequency=1000, recency=10, T=100)
        np.seterr(**old_settings)
        assert p_alive == 0.

    def test_using_weights_col_gives_correct_results(self, cdnow_customers):
        cdnow_customers_weights = cdnow_customers.copy()
        cdnow_customers_weights['weights'] = 1.0
        cdnow_customers_weights = cdnow_customers_weights.groupby(['frequency', 'recency', 'T'])['weights'].sum()
        cdnow_customers_weights = cdnow_customers_weights.reset_index()
        assert (cdnow_customers_weights['weights'] > 1).any()

        bgf_weights = estimation.BetaGeoFitter(penalizer_coef=0.0)
        bgf_weights.fit(
            cdnow_customers_weights['frequency'],
            cdnow_customers_weights['recency'],
            cdnow_customers_weights['T'],
            weights=cdnow_customers_weights['weights']
        )


        bgf_no_weights = estimation.BetaGeoFitter(penalizer_coef=0.0)
        bgf_no_weights.fit(
            cdnow_customers['frequency'],
            cdnow_customers['recency'],
            cdnow_customers['T']
        )

        npt.assert_almost_equal(
            np.array(bgf_no_weights._unload_params('r', 'alpha', 'a', 'b')),
            np.array(bgf_weights._unload_params('r', 'alpha', 'a', 'b')),
        decimal=4)

class TestModifiedBetaGammaFitter():

    def test_sum_of_scalar_inputs_to_negative_log_likelihood_is_equal_to_array(self):
        mbgf = estimation.ModifiedBetaGeoFitter
        x = np.array([1, 3])
        t_x = np.array([2, 2])
        t = np.array([5, 6])
        weights=np.array([1, 1])
        params = [1, 1, 1, 1]
        assert mbgf._negative_log_likelihood(params, np.array([x[0]]), np.array([t_x[0]]), np.array([t[0]]), weights[0], 0) \
            + mbgf._negative_log_likelihood(params, np.array([x[1]]), np.array([t_x[1]]), np.array([t[1]]), weights[0], 0) \
            == 2 * mbgf._negative_log_likelihood(params, x, t_x, t, weights, 0)

    def test_params_out_is_close_to_BTYDplus(self, cdnow_customers):
        """ See https://github.com/mplatzer/BTYDplus """
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=3)
        expected = np.array([0.525, 6.183, 0.891, 1.614])
        npt.assert_array_almost_equal(expected, np.array(mbfg._unload_params('r', 'alpha', 'a', 'b')), decimal=3)

    def test_conditional_expectation_returns_same_value_as_Hardie_excel_sheet(self, cdnow_customers):
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        x = 2
        t_x = 30.43
        T = 38.86
        t = 39
        expected = 1.226
        actual = mbfg.conditional_expected_number_of_purchases_up_to_time(t, x, t_x, T)
        assert abs(expected - actual) < 0.05

    def test_expectation_returns_same_value_Hardie_excel_sheet(self, cdnow_customers):
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], tol=1e-6, iterative_fitting=3)

        times = np.array([0.1429, 1.0, 3.00, 31.8571, 32.00, 78.00])
        expected = np.array([0.0078, 0.0532, 0.1506, 1.0405, 1.0437, 1.8576])
        actual = mbfg.expected_number_of_purchases_up_to_time(times)
        npt.assert_allclose(actual, expected, rtol=0.05)

    def test_conditional_probability_alive_returns_lessthan_1_if_no_repeat_purchases(self, cdnow_customers):
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        assert mbfg.conditional_probability_alive(0, 1, 1) < 1.0


    def test_conditional_probability_alive_is_between_0_and_1(self, cdnow_customers):
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                for k in range(j, 100, 10):
                    assert 0 <= mbfg.conditional_probability_alive(i, j, k) <= 1.0

    def test_fit_method_allows_for_better_accuracy_by_using_iterative_fitting(self, cdnow_customers):
        mbfg1 = estimation.ModifiedBetaGeoFitter()
        mbfg2 = estimation.ModifiedBetaGeoFitter()

        np.random.seed(0)
        mbfg1.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        np.random.seed(0)
        mbfg2.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=5)
        assert mbfg1._negative_log_likelihood_ >= mbfg2._negative_log_likelihood_

    def test_penalizer_term_will_shrink_coefs_to_0(self, cdnow_customers):
        mbfg_no_penalizer = estimation.ModifiedBetaGeoFitter()
        mbfg_no_penalizer.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        params_1 = np.array(list(mbfg_no_penalizer.params_.values()))

        mbfg_with_penalizer = estimation.ModifiedBetaGeoFitter(penalizer_coef=0.1)
        mbfg_with_penalizer.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=3)
        params_2 = np.array(list(mbfg_with_penalizer.params_.values()))
        assert params_2.sum() < params_1.sum()

        mbfg_with_more_penalizer = estimation.ModifiedBetaGeoFitter(penalizer_coef=1.)
        mbfg_with_more_penalizer.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=5)
        params_3 = np.array(list(mbfg_with_more_penalizer.params_.values()))
        assert params_3.sum() < params_2.sum()

    def test_conditional_probability_alive_matrix(self, cdnow_customers):
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        Z = mbfg.conditional_probability_alive_matrix()
        max_t = int(mbfg.data['T'].max())

        for t_x in range(Z.shape[0]):
            for x in range(Z.shape[1]):
                assert Z[t_x][x] == mbfg.conditional_probability_alive(x, t_x, max_t)

    def test_probability_of_n_purchases_up_to_time_same_as_R_BTYD(self):
        """ See https://cran.r-project.org/web/packages/BTYD/BTYD.pdf """
        mbgf = estimation.ModifiedBetaGeoFitter()
        mbgf.params_ = OrderedDict({'r':0.243, 'alpha':4.414, 'a':0.793, 'b':2.426})
        # probability that a customer will make 10 repeat transactions in the
        # time interval (0,2]
        expected = 1.07869e-07
        actual = mbgf.probability_of_n_purchases_up_to_time(2, 10)
        assert abs(expected - actual) < 10e-5
        # PMF
        expected = np.array([0.0019995214, 0.0015170236, 0.0011633150, 0.0009003148, 0.0007023638,
                             0.0005517902, 0.0004361913, 0.0003467171, 0.0002769613, 0.0002222260])
        actual = np.array([mbgf.probability_of_n_purchases_up_to_time(30, n) for n in range(11, 21)])
        npt.assert_allclose(expected, actual, rtol=0.5)

    def test_scaling_inputs_gives_same_or_similar_results(self, cdnow_customers):
        mbgf = estimation.ModifiedBetaGeoFitter()
        mbgf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        scale = 10.
        mbgf_with_large_inputs = estimation.ModifiedBetaGeoFitter()
        mbgf_with_large_inputs.fit(cdnow_customers['frequency'], scale * cdnow_customers['recency'], scale * cdnow_customers['T'], iterative_fitting=2)
        assert mbgf_with_large_inputs._scale < 1.

        assert abs(mbgf_with_large_inputs.conditional_probability_alive(1, scale * 1, scale * 2) - mbgf.conditional_probability_alive(1, 1, 2)) < 10e-2
        assert abs(mbgf_with_large_inputs.conditional_probability_alive(1, scale * 2, scale * 10) - mbgf.conditional_probability_alive(1, 2, 10)) < 10e-2

    def test_mgbf_does_not_hang_for_small_datasets_but_can_be_improved_with_iterative_fitting(self, cdnow_customers):
        reduced_dataset = cdnow_customers.iloc[:2]
        mbfg1 = estimation.ModifiedBetaGeoFitter()
        mbfg2 = estimation.ModifiedBetaGeoFitter()

        np.random.seed(0)
        mbfg1.fit(reduced_dataset['frequency'], reduced_dataset['recency'], reduced_dataset['T'])

        np.random.seed(0)
        mbfg2.fit(reduced_dataset['frequency'], reduced_dataset['recency'], reduced_dataset['T'], iterative_fitting=10)
        assert mbfg1._negative_log_likelihood_ >= mbfg2._negative_log_likelihood_

    def test_purchase_predictions_do_not_differ_much_if_looking_at_hourly_or_daily_frequencies(self):
        transaction_data = load_transaction_data(parse_dates=['date'])
        daily_summary = utils.summary_data_from_transaction_data(transaction_data, 'id', 'date', observation_period_end=max(transaction_data.date), freq='D')
        hourly_summary = utils.summary_data_from_transaction_data(transaction_data, 'id', 'date', observation_period_end=max(transaction_data.date), freq='h')
        thirty_days = 30
        hours_in_day = 24
        mbfg = estimation.ModifiedBetaGeoFitter()

        np.random.seed(0)
        mbfg.fit(daily_summary['frequency'], daily_summary['recency'], daily_summary['T'])
        thirty_day_prediction_from_daily_data = mbfg.expected_number_of_purchases_up_to_time(thirty_days)

        np.random.seed(0)
        mbfg.fit(hourly_summary['frequency'], hourly_summary['recency'], hourly_summary['T'])
        thirty_day_prediction_from_hourly_data = mbfg.expected_number_of_purchases_up_to_time(thirty_days * hours_in_day)

        npt.assert_almost_equal(thirty_day_prediction_from_daily_data, thirty_day_prediction_from_hourly_data)

    def test_fit_with_index(self, cdnow_customers):
        mbgf = estimation.ModifiedBetaGeoFitter()
        index = range(len(cdnow_customers), 0, -1)
        mbgf.fit(
            cdnow_customers['frequency'],
            cdnow_customers['recency'],
            cdnow_customers['T'],
            index=index
        )
        assert (mbgf.data.index == index).all() == True

        mbgf = estimation.ModifiedBetaGeoFitter()
        mbgf.fit(
            cdnow_customers['frequency'],
            cdnow_customers['recency'],
            cdnow_customers['T'],
            index=None
        )
        assert (mbgf.data.index == index).all() == False
