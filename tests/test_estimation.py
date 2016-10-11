from __future__ import print_function

import numpy as np
import pandas as pd
import numpy.testing as npt

import lifetimes.estimation as estimation
import lifetimes.utils as utils
from lifetimes.datasets import load_cdnow, load_summary_data_with_monetary_value, load_donations

cdnow_customers = load_cdnow()
cdnow_customers_with_monetary_value = load_summary_data_with_monetary_value()
donations = load_donations()

class TestBetaGeoBetaBinomFitter():

    def test_params_out_is_close_to_Hardie_paper(self):

        bbtf = estimation.BetaGeoBetaBinomFitter()
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['n'],
            donations['n_custs'],
        )
        expected = np.array([1.204, 0.750, 0.657, 2.783])
        npt.assert_array_almost_equal(expected, np.array(bbtf._unload_params('alpha','beta','gamma','delta')),
                                      decimal=2)

    def test_prob_alive_is_close_to_Hardie_paper_table_6(self):
        """Table 6: P(Alive in 2002) as a Function of Recency and Frequency"""

        bbtf = estimation.BetaGeoBetaBinomFitter()
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['n'],
            donations['n_custs'],
        )

        bbtf.data['prob_alive'] = bbtf.conditional_probability_alive(1)

        # Expected probabilities for last year 1995-0 repeat, 1999-2 repeat, 2001-6 repeat
        expected = np.array([0.11, 0.59, 0.93])
        prob_list = np.zeros(3)
        prob_list[0] = (bbtf.data[(bbtf.data['frequency'] == 0) & (bbtf.data['recency'] == 0)]['prob_alive'])
        prob_list[1] = (bbtf.data[(bbtf.data['frequency'] == 2) & (bbtf.data['recency'] == 4)]['prob_alive'])
        prob_list[2] = (bbtf.data[(bbtf.data['frequency'] == 6) & (bbtf.data['recency'] == 6)]['prob_alive'])
        npt.assert_array_almost_equal(expected, prob_list, decimal=2)

    def test_conditional_expectation_returns_same_value_as_Hardie_excel_sheet(self):
        """
        Total from Hardie's Conditional Expectations (II) sheet.

        http://brucehardie.com/notes/010/BGBB_2011-01-20_XLSX.zip

        """

        bbtf = estimation.BetaGeoBetaBinomFitter()
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['n'],
            donations['n_custs'],
        )
        pred_purchases = bbtf.conditional_expected_number_of_purchases_up_to_time(5) * donations['n_custs']
        expected = 12884.2 # Sum of column F Exp Tot
        npt.assert_almost_equal(expected, pred_purchases.sum(), decimal=0)

    def test_expected_purchases_in_n_periods_returns_same_value_as_Hardie_excel_sheet(selfself):
        """Total expected from Hardie's In-Sample Fit sheet."""

        bbtf = estimation.BetaGeoBetaBinomFitter()
        bbtf.fit(
            donations['frequency'],
            donations['recency'],
            donations['n'],
            donations['n_custs'],
        )
        expected = np.array([3454.9, 1253.1]) # Cells C18 and C24
        estimated = bbtf.expected_number_of_transactions_in_first_n_periods(6).loc[[0,6]].values.flatten()
        npt.assert_almost_equal(expected, estimated, decimal=0)


class TestGammaGammaFitter():

    def test_params_out_is_close_to_Hardie_paper(self):
        ggf = estimation.GammaGammaFitter()
        ggf.fit(
            cdnow_customers_with_monetary_value['frequency'],
            cdnow_customers_with_monetary_value['monetary_value'],
            iterative_fitting=3
        )
        expected = np.array([6.25, 3.74, 15.44])
        npt.assert_array_almost_equal(expected, np.array(ggf._unload_params('p', 'q', 'v')), decimal=2)

    def test_conditional_expected_average_profit(self):
        from collections import OrderedDict

        ggf = estimation.GammaGammaFitter()
        ggf.params_ = OrderedDict({'p':6.25, 'q':3.74, 'v':15.44})

        summary = cdnow_customers_with_monetary_value.head(10)
        estimates = ggf.conditional_expected_average_profit(summary['frequency'], summary['monetary_value'])
        expected = np.array([24.65, 18.91, 35.17, 35.17, 35.17, 71.46, 18.91, 35.17, 27.28, 35.17]) # from Hardie spreadsheet http://brucehardie.com/notes/025/

        npt.assert_allclose(estimates.values, expected, atol=0.1)

    def test_customer_lifetime_value_with_bgf(self):
        from collections import OrderedDict

        ggf = estimation.GammaGammaFitter()
        ggf.params_ = OrderedDict({'p':6.25, 'q':3.74, 'v':15.44})

        bgf = estimation.BetaGeoFitter()
        bgf.fit(cdnow_customers_with_monetary_value['frequency'], cdnow_customers_with_monetary_value['recency'], cdnow_customers_with_monetary_value['T'], iterative_fitting=3)

        ggf_clv = ggf.customer_lifetime_value(
                bgf,
                cdnow_customers_with_monetary_value['frequency'],
                cdnow_customers_with_monetary_value['recency'],
                cdnow_customers_with_monetary_value['T'],
                cdnow_customers_with_monetary_value['monetary_value']
        )

        utils_clv = utils.customer_lifetime_value(
                bgf,
                cdnow_customers_with_monetary_value['frequency'],
                cdnow_customers_with_monetary_value['recency'],
                cdnow_customers_with_monetary_value['T'],
                ggf.conditional_expected_average_profit(cdnow_customers_with_monetary_value['frequency'],cdnow_customers_with_monetary_value['monetary_value'])
        )
        npt.assert_equal(ggf_clv.values, utils_clv.values)


class TestParetoNBDFitter():

    def test_overflow_error(self):
         
        ptf = estimation.ParetoNBDFitter()
        params = np.array([10.465, 7.98565181e-03, 3.0516, 2.820])
        freq = np.array([400., 500., 500.] )
        rec = np.array([5., 1., 4.])
        age = np.array([6., 37., 37.])
        assert all([r < 0 and not np.isinf(r) and not pd.isnull(r) 
                    for r in ptf._log_A_0(params, freq, rec, age)])


    def test_sum_of_scalar_inputs_to_negative_log_likelihood_is_equal_to_array(self):
        ptf = estimation.ParetoNBDFitter
        x = np.array([1,3])
        t_x = np.array([2,2])
        t = np.array([5,6])
        params = [1,1,1,1]
        assert ptf()._negative_log_likelihood(params, np.array([x[0]]), np.array([t_x[0]]), np.array([t[0]]), 0) \
             + ptf()._negative_log_likelihood(params, np.array([x[1]]), np.array([t_x[1]]), np.array([t[1]]), 0) \
            == ptf()._negative_log_likelihood(params, x, t_x, t, 0)

    def test_params_out_is_close_to_Hardie_paper(self):
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=3)
        expected = np.array([ 0.553, 10.578, 0.606, 11.669])
        npt.assert_array_almost_equal(expected, np.array(ptf._unload_params('r', 'alpha', 's', 'beta')), decimal=3)

    def test_expectation_returns_same_value_as_R_BTYD(self):
        """ From https://cran.r-project.org/web/packages/BTYD/BTYD.pdf """
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], tol=1e-6)

        expected = np.array([0.00000000, 0.05077821, 0.09916088, 0.14542507, 0.18979930,
            0.23247466, 0.27361274, 0.31335159, 0.35181024, 0.38909211])
        actual = ptf.expected_number_of_purchases_up_to_time(range(10))
        npt.assert_allclose(expected, actual, atol=0.01)

    def test_conditional_expectation_returns_same_value_as_R_BTYD(self):
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

    def test_conditional_probability_alive_is_between_0_and_1(self):
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        for freq in np.arange(0, 100, 10.):
            for recency in np.arange(0, 100, 10.):
                for t in np.arange(recency, 100, 10.):
                    assert 0.0 <= ptf.conditional_probability_alive(freq, recency, t) <= 1.0

    def test_conditional_probability_alive_matrix(self):
        ptf = estimation.ParetoNBDFitter()
        ptf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        Z = ptf.conditional_probability_alive_matrix()
        max_t = int(ptf.data['T'].max())

        for t_x in range(Z.shape[0]):
            for x in range(Z.shape[1]):
                assert Z[t_x][x] == ptf.conditional_probability_alive(x, t_x, max_t)


class TestBetaGammaFitter():

    def test_sum_of_scalar_inputs_to_negative_log_likelihood_is_equal_to_array(self):
        bgf = estimation.BetaGeoFitter
        x = np.array([1,3])
        t_x = np.array([2,2])
        t = np.array([5,6])
        params = [1,1,1,1]
        assert bgf._negative_log_likelihood(params, np.array([x[0]]), np.array([t_x[0]]), np.array([t[0]]), 0) \
             + bgf._negative_log_likelihood(params, np.array([x[1]]), np.array([t_x[1]]), np.array([t[1]]), 0) \
            == bgf._negative_log_likelihood(params, x, t_x, t, 0)
 
    def test_params_out_is_close_to_Hardie_paper(self):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=3)
        expected = np.array([0.243, 4.414, 0.793, 2.426])
        npt.assert_array_almost_equal(expected, np.array(bfg._unload_params('r', 'alpha', 'a', 'b')), decimal=3)

    def test_conditional_expectation_returns_same_value_as_Hardie_excel_sheet(self):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        x = 2
        t_x = 30.43
        T = 38.86
        t = 39 
        expected = 1.226
        actual = bfg.conditional_expected_number_of_purchases_up_to_time(t, x, t_x, T) 
        assert abs(expected - actual) < 0.001

    def test_expectation_returns_same_value_Hardie_excel_sheet(self):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], tol=1e-6)

        times = np.array([0.1429, 1.0, 3.00, 31.8571, 32.00, 78.00])
        expected = np.array([0.0078 ,0.0532 ,0.1506 ,1.0405,1.0437, 1.8576])
        actual = bfg.expected_number_of_purchases_up_to_time(times)
        npt.assert_array_almost_equal(actual, expected, decimal=3)

    def test_conditional_probability_alive_returns_1_if_no_repeat_purchases(self):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        assert bfg.conditional_probability_alive(0, 1, 1) == 1.0


    def test_conditional_probability_alive_is_between_0_and_1(self):
        bfg = estimation.BetaGeoFitter()
        bfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                for k in range(j, 100, 10):
                    assert 0 <= bfg.conditional_probability_alive(i, j, k) <= 1.0


    def test_fit_method_allows_for_better_accuracy_by_using_iterative_fitting(self):
        bfg1 = estimation.BetaGeoFitter()
        bfg2 = estimation.BetaGeoFitter()

        np.random.seed(0)
        bfg1.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        np.random.seed(0)
        bfg2.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=5)
        assert bfg1._negative_log_likelihood_ >= bfg2._negative_log_likelihood_


    def test_penalizer_term_will_shrink_coefs_to_0(self):
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


    def test_conditional_probability_alive_matrix(self):
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
        from collections import OrderedDict
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


    def test_scaling_inputs_gives_same_or_similar_results(self):
        bgf = estimation.BetaGeoFitter()
        bgf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        scale = 10
        bgf_with_large_inputs = estimation.BetaGeoFitter()
        bgf_with_large_inputs.fit(cdnow_customers['frequency'], scale*cdnow_customers['recency'], scale*cdnow_customers['T'])
        assert bgf_with_large_inputs._scale < 1.

        assert abs(bgf_with_large_inputs.conditional_probability_alive(1, scale*1, scale*2) - bgf.conditional_probability_alive(1, 1, 2)) < 10e-5
        assert abs(bgf_with_large_inputs.conditional_probability_alive(1, scale*2, scale*10) - bgf.conditional_probability_alive(1, 2, 10)) < 10e-5


class TestModifiedBetaGammaFitter():

    def test_sum_of_scalar_inputs_to_negative_log_likelihood_is_equal_to_array(self):
        mbgf = estimation.ModifiedBetaGeoFitter
        x = np.array([1, 3])
        t_x = np.array([2, 2])
        t = np.array([5, 6])
        params = [1, 1, 1, 1]
        assert mbgf._negative_log_likelihood(params, np.array([x[0]]), np.array([t_x[0]]), np.array([t[0]]), 0) \
             + mbgf._negative_log_likelihood(params, np.array([x[1]]), np.array([t_x[1]]), np.array([t[1]]), 0) \
            == mbgf._negative_log_likelihood(params, x, t_x, t, 0)
 
    def test_params_out_is_close_to_BTYDplus(self):
        """ See https://github.com/mplatzer/BTYDplus """
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=3)
        expected = np.array([0.525, 6.183, 0.891, 1.614])
        npt.assert_array_almost_equal(expected, np.array(mbfg._unload_params('r', 'alpha', 'a', 'b')), decimal=3)

    def test_conditional_expectation_returns_same_value_as_Hardie_excel_sheet(self):
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        x = 2
        t_x = 30.43
        T = 38.86
        t = 39
        expected = 1.226
        actual = mbfg.conditional_expected_number_of_purchases_up_to_time(t, x, t_x, T) 
        assert abs(expected - actual) < 0.05

    def test_expectation_returns_same_value_Hardie_excel_sheet(self):
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], tol=1e-6)

        times = np.array([0.1429, 1.0, 3.00, 31.8571, 32.00, 78.00])
        expected = np.array([0.0078, 0.0532, 0.1506, 1.0405, 1.0437, 1.8576])
        actual = mbfg.expected_number_of_purchases_up_to_time(times)
        npt.assert_allclose(actual, expected, rtol=0.05) 

    def test_conditional_probability_alive_returns_lessthan_1_if_no_repeat_purchases(self):
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        assert mbfg.conditional_probability_alive(0, 1, 1) < 1.0


    def test_conditional_probability_alive_is_between_0_and_1(self):
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                for k in range(j, 100, 10):
                    assert 0 <= mbfg.conditional_probability_alive(i, j, k) <= 1.0


    def test_fit_method_allows_for_better_accuracy_by_using_iterative_fitting(self):
        mbfg1 = estimation.ModifiedBetaGeoFitter()
        mbfg2 = estimation.ModifiedBetaGeoFitter()

        np.random.seed(0)
        mbfg1.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])

        np.random.seed(0)
        mbfg2.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'], iterative_fitting=5)
        assert mbfg1._negative_log_likelihood_ >= mbfg2._negative_log_likelihood_


    def test_penalizer_term_will_shrink_coefs_to_0(self):
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


    def test_conditional_probability_alive_matrix(self):
        mbfg = estimation.ModifiedBetaGeoFitter()
        mbfg.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        Z = mbfg.conditional_probability_alive_matrix()
        max_t = int(mbfg.data['T'].max())

        for t_x in range(Z.shape[0]):
            for x in range(Z.shape[1]):
                assert Z[t_x][x] == mbfg.conditional_probability_alive(x, t_x, max_t)


    def test_probability_of_n_purchases_up_to_time_same_as_R_BTYD(self):
        """ See https://cran.r-project.org/web/packages/BTYD/BTYD.pdf """
        from collections import OrderedDict
        mbgf = estimation.ModifiedBetaGeoFitter()
        mbgf.params_ = OrderedDict({'r':0.243, 'alpha':4.414, 'a':0.793, 'b':2.426})
        # probability that a customer will make 10 repeat transactions in the
        # time interval (0,2]
        expected = 1.07869e-07
        actual = mbgf.probability_of_n_purchases_up_to_time(2,10)
        assert abs(expected - actual) < 10e-5
        # PMF
        expected = np.array([0.0019995214, 0.0015170236, 0.0011633150, 0.0009003148, 0.0007023638,
                             0.0005517902, 0.0004361913, 0.0003467171, 0.0002769613, 0.0002222260])
        actual = np.array([mbgf.probability_of_n_purchases_up_to_time(30,n) for n in range(11,21)])
        npt.assert_allclose(expected, actual, rtol=0.5)


    def test_scaling_inputs_gives_same_or_similar_results(self):
        mbgf = estimation.ModifiedBetaGeoFitter()
        mbgf.fit(cdnow_customers['frequency'], cdnow_customers['recency'], cdnow_customers['T'])
        scale = 10.
        mbgf_with_large_inputs = estimation.ModifiedBetaGeoFitter()
        mbgf_with_large_inputs.fit(cdnow_customers['frequency'], scale*cdnow_customers['recency'], scale*cdnow_customers['T'])
        assert mbgf_with_large_inputs._scale < 1.

        assert abs(mbgf_with_large_inputs.conditional_probability_alive(1, scale*1, scale*2) - mbgf.conditional_probability_alive(1, 1, 2)) < 10e-2
        assert abs(mbgf_with_large_inputs.conditional_probability_alive(1, scale*2, scale*10) - mbgf.conditional_probability_alive(1, 2, 10)) < 10e-2
