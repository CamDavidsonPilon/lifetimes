import pytest

import matplotlib
matplotlib.use('AGG')  # use a non-interactive backend
from matplotlib import pyplot as plt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from lifetimes import plotting
from lifetimes import BetaGeoFitter, ParetoNBDFitter, ModifiedBetaGeoFitter
from lifetimes.datasets import (load_cdnow_summary, load_transaction_data,
                                load_dataset)
from lifetimes import utils


@pytest.fixture()
def cd_data():
    return load_cdnow_summary()


@pytest.fixture()
def bgf(cd_data):
    bgf_model = BetaGeoFitter()
    bgf_model.fit(cd_data['frequency'], cd_data['recency'], cd_data['T'], iterative_fitting=1)
    return bgf_model


@pytest.fixture()
def transaction_data():
    return load_transaction_data()


@pytest.fixture()
def cdnow_transactions():
    transactions = load_dataset('CDNOW_sample.txt', header=None, sep='\s+')
    transactions.columns = ['id_total', 'id_sample', 'date', 'num_cd_purc',
                            'total_value']
    return transactions[['id_sample', 'date']]


@pytest.fixture()
def bgf_transactions(cdnow_transactions):
    transactions_summary = utils.summary_data_from_transaction_data(
        cdnow_transactions, 'id_sample', 'date', datetime_format='%Y%m%d',
        observation_period_end='19970930', freq='W')

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(transactions_summary['frequency'],
            transactions_summary['recency'], transactions_summary['T'])
    return bgf


@pytest.mark.plottest
class TestPlotting():

    @classmethod
    def setup_class(cls):
        np.random.seed(123456789)  # static random seed for this test class

    def test_plot_period_transactions(self, bgf):
        expected = [1411, 439, 214, 100, 62, 38, 29, 1411, 439, 214, 100, 62, 38, 29]

        ax = plotting.plot_period_transactions(bgf)

        assert_allclose([p.get_height() for p in ax.patches], expected, rtol=0.3)
        assert_equal(ax.title.get_text(), "Frequency of Repeat Transactions")
        assert_equal(ax.xaxis.get_label().get_text(), "Number of Calibration Period Transactions")
        assert_equal(ax.yaxis.get_label().get_text(), "Customers")
        assert_array_equal([label.get_text() for label in ax.legend_.get_texts()], ["Actual", "Model"])
        plt.close()

    def test_plot_period_transactions_mbgf(self, cd_data):

        mbgf = ModifiedBetaGeoFitter()
        mbgf.fit(cd_data['frequency'], cd_data['recency'], cd_data['T'], iterative_fitting=1)

        ax = plotting.plot_period_transactions(mbgf)

        assert_equal(ax.title.get_text(), "Frequency of Repeat Transactions")
        assert_equal(ax.xaxis.get_label().get_text(), "Number of Calibration Period Transactions")
        assert_equal(ax.yaxis.get_label().get_text(), "Customers")
        assert_array_equal([label.get_text() for label in ax.legend_.get_texts()], ["Actual", "Model"])
        plt.close()

    def test_plot_period_transactions_max_frequency(self, bgf):
        expected = [1411, 439, 214, 100, 62, 38, 29, 23, 7, 5, 5, 5,
                    1429, 470, 155, 89, 71, 39, 26, 20, 18, 9, 6, 7]

        ax = plotting.plot_period_transactions(bgf, max_frequency=12)

        assert_allclose([p.get_height() for p in ax.patches], expected, atol=50)  # can be large relative differences for small counts
        assert_equal(ax.title.get_text(), "Frequency of Repeat Transactions")
        assert_equal(ax.xaxis.get_label().get_text(), "Number of Calibration Period Transactions")
        assert_equal(ax.yaxis.get_label().get_text(), "Customers")
        assert_array_equal([label.get_text() for label in ax.legend_.get_texts()], ["Actual", "Model"])
        plt.close()

    def test_plot_period_transactions_labels(self, bgf):
        expected = [1411, 439, 214, 100, 62, 38, 29, 1411, 439, 214, 100, 62, 38, 29]

        ax = plotting.plot_period_transactions(bgf, label=['A', 'B'])

        assert_allclose([p.get_height() for p in ax.patches], expected, rtol=0.3)
        assert_equal(ax.title.get_text(), "Frequency of Repeat Transactions")
        assert_equal(ax.xaxis.get_label().get_text(), "Number of Calibration Period Transactions")
        assert_equal(ax.yaxis.get_label().get_text(), "Customers")
        assert_array_equal([label.get_text() for label in ax.legend_.get_texts()], ["A", "B"])
        plt.close()

    def test_plot_frequency_recency_matrix(self, bgf):
        shape = (39, 30)
        row_idx = 29
        row = [0.005, 0.020, 0.037, 0.054, 0.070, 0.085, 0.099, 0.110, 0.120, 0.127, 0.133,
               0.136, 0.136, 0.135, 0.131, 0.125, 0.119, 0.111, 0.102, 0.093, 0.084, 0.075,
               0.066, 0.058, 0.050, 0.044, 0.038, 0.032, 0.027, 0.023]

        ax = plotting.plot_frequency_recency_matrix(bgf)
        ar = ax.get_images()[0].get_array()
        assert_array_equal(ar.shape, shape)
        assert_allclose(ar[row_idx, :].data, row, atol=0.01)  # only test one row for brevity
        assert_equal(ax.title.get_text(), "Expected Number of Future Purchases for 1 Unit of Time,\nby Frequency and Recency of a Customer")
        assert_equal(ax.xaxis.get_label().get_text(), "Customer's Historical Frequency")
        assert_equal(ax.yaxis.get_label().get_text(), "Customer's Recency")
        plt.close()

    def test_plot_frequency_recency_matrix_max_recency(self, bgf):
        shape = (101, 30)
        col_idx = 25
        col = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.001, 0.002,
               0.002, 0.004, 0.005, 0.007, 0.010, 0.014, 0.018, 0.024, 0.032, 0.041, 0.052,
               0.065, 0.080, 0.096, 0.112, 0.129, 0.145, 0.160, 0.174, 0.186, 0.196, 0.205,
               0.212, 0.218, 0.222, 0.226, 0.229, 0.232, 0.233]

        ax = plotting.plot_frequency_recency_matrix(bgf, max_recency=100)
        ar = ax.get_images()[0].get_array()
        assert_array_equal(ar.shape, shape)
        assert_allclose(ar[:, col_idx].data, col, atol=0.01)  # only test one row for brevity
        assert_equal(ax.title.get_text(), "Expected Number of Future Purchases for 1 Unit of Time,\nby Frequency and Recency of a Customer")
        assert_equal(ax.xaxis.get_label().get_text(), "Customer's Historical Frequency")
        assert_equal(ax.yaxis.get_label().get_text(), "Customer's Recency")
        plt.close()

    def test_plot_frequency_recency_matrix_max_frequency(self, bgf):
        shape = (39, 101)
        row_idx = 35
        row = [0.005, 0.021, 0.041, 0.061, 0.082, 0.103, 0.125, 0.146, 0.167, 0.188, 0.208,
               0.229, 0.250, 0.270, 0.290, 0.310, 0.330, 0.349, 0.369, 0.388, 0.406, 0.425,
               0.443, 0.460, 0.478, 0.495, 0.511, 0.528, 0.543, 0.559, 0.573, 0.587, 0.601,
               0.614, 0.627, 0.639, 0.650, 0.660, 0.670, 0.679, 0.688, 0.695, 0.702, 0.708,
               0.713, 0.718, 0.721, 0.724, 0.726, 0.727, 0.727, 0.726, 0.724, 0.721, 0.718,
               0.713, 0.708, 0.702, 0.695, 0.687, 0.679, 0.670, 0.660, 0.649, 0.638, 0.627,
               0.615, 0.602, 0.589, 0.575, 0.562, 0.548, 0.533, 0.519, 0.504, 0.489, 0.475,
               0.460, 0.445, 0.430, 0.416, 0.401, 0.387, 0.372, 0.359, 0.345, 0.331, 0.318,
               0.305, 0.293, 0.280, 0.269, 0.257, 0.246, 0.235, 0.224, 0.214, 0.204, 0.195,
               0.186, 0.177]

        ax = plotting.plot_frequency_recency_matrix(bgf, max_frequency=100)
        ar = ax.get_images()[0].get_array()
        assert_array_equal(ar.shape, shape)
        assert_allclose(ar[row_idx, :].data, row, atol=0.01)  # only test one row for brevity
        assert_equal(ax.title.get_text(), "Expected Number of Future Purchases for 1 Unit of Time,\nby Frequency and Recency of a Customer")
        assert_equal(ax.xaxis.get_label().get_text(), "Customer's Historical Frequency")
        assert_equal(ax.yaxis.get_label().get_text(), "Customer's Recency")
        plt.close()

    def test_plot_frequency_recency_matrix_max_frequency_max_recency(self, bgf):
        shape = (101, 101)
        row_idx = 95
        row = [0.002, 0.008, 0.017, 0.025, 0.034, 0.043, 0.052, 0.060, 0.069, 0.078, 0.087,
               0.096, 0.105, 0.114, 0.123, 0.132, 0.140, 0.149, 0.158, 0.166, 0.175, 0.184,
               0.192, 0.201, 0.209, 0.218, 0.226, 0.235, 0.243, 0.251, 0.259, 0.267, 0.275,
               0.283, 0.291, 0.299, 0.307, 0.314, 0.322, 0.330, 0.337, 0.344, 0.352, 0.359,
               0.366, 0.373, 0.379, 0.386, 0.393, 0.399, 0.405, 0.411, 0.417, 0.423, 0.429,
               0.435, 0.440, 0.445, 0.450, 0.455, 0.460, 0.465, 0.469, 0.473, 0.477, 0.481,
               0.484, 0.488, 0.491, 0.494, 0.497, 0.499, 0.501, 0.503, 0.505, 0.506, 0.508,
               0.509, 0.509, 0.510, 0.510, 0.510, 0.510, 0.509, 0.508, 0.507, 0.506, 0.504,
               0.503, 0.501, 0.498, 0.496, 0.493, 0.490, 0.486, 0.483, 0.479, 0.475, 0.471,
               0.466, 0.462]

        ax = plotting.plot_frequency_recency_matrix(bgf, max_frequency=100, max_recency=100)
        ar = ax.get_images()[0].get_array()
        assert_array_equal(ar.shape, shape)
        assert_allclose(ar[row_idx, :].data, row, atol=0.01)  # only test one row for brevity
        assert_equal(ax.title.get_text(), "Expected Number of Future Purchases for 1 Unit of Time,\nby Frequency and Recency of a Customer")
        assert_equal(ax.xaxis.get_label().get_text(), "Customer's Historical Frequency")
        assert_equal(ax.yaxis.get_label().get_text(), "Customer's Recency")
        plt.close()

    def test_plot_probability_alive_matrix(self, bgf):
        shape = (39, 30)
        row_idx = 35
        row = [1.0, 0.736, 0.785, 0.814, 0.833, 0.846, 0.855, 0.862, 0.866, 0.869, 0.871,
               0.872, 0.873, 0.873, 0.872, 0.871, 0.869, 0.867, 0.865, 0.862, 0.859, 0.856,
               0.852, 0.848, 0.844, 0.839, 0.834, 0.829, 0.823, 0.817]

        ax = plotting.plot_probability_alive_matrix(bgf)
        ar = ax.get_images()[0].get_array()
        assert_array_equal(ar.shape, shape)
        assert_allclose(ar[row_idx, :].data, row, atol=0.01)  # only test one row for brevity
        assert_equal(ax.title.get_text(), "Probability Customer is Alive,\nby Frequency and Recency of a Customer")
        assert_equal(ax.xaxis.get_label().get_text(), "Customer's Historical Frequency")
        assert_equal(ax.yaxis.get_label().get_text(), "Customer's Recency")
        plt.close()

    def test_plot_probability_alive_matrix_max_frequency(self, bgf):
        shape = (39, 101)
        row_idx = 35
        row = [1.0, 0.736, 0.785, 0.814, 0.833, 0.846, 0.855, 0.862, 0.866, 0.869, 0.871,
               0.872, 0.873, 0.873, 0.872, 0.871, 0.869, 0.867, 0.865, 0.862, 0.859, 0.856,
               0.852, 0.848, 0.844, 0.839, 0.834, 0.829, 0.823, 0.817, 0.811, 0.805, 0.798,
               0.791, 0.783, 0.775, 0.767, 0.759, 0.750, 0.741, 0.731, 0.721, 0.711, 0.701,
               0.690, 0.679, 0.667, 0.656, 0.644, 0.631, 0.619, 0.606, 0.593, 0.580, 0.566,
               0.552, 0.539, 0.525, 0.511, 0.496, 0.482, 0.468, 0.454, 0.439, 0.425, 0.411,
               0.397, 0.383, 0.369, 0.355, 0.342, 0.329, 0.316, 0.303, 0.290, 0.278, 0.266,
               0.254, 0.243, 0.232, 0.221, 0.211, 0.201, 0.191, 0.182, 0.173, 0.164, 0.156,
               0.148, 0.140, 0.133, 0.126, 0.119, 0.113, 0.106, 0.101, 0.095, 0.090, 0.085,
               0.080, 0.075]

        ax = plotting.plot_probability_alive_matrix(bgf, max_frequency=100)
        ar = ax.get_images()[0].get_array()
        assert_array_equal(ar.shape, shape)
        assert_allclose(ar[row_idx, :].data, row, atol=0.01)  # only test one row for brevity
        assert_equal(ax.title.get_text(), "Probability Customer is Alive,\nby Frequency and Recency of a Customer")
        assert_equal(ax.xaxis.get_label().get_text(), "Customer's Historical Frequency")
        assert_equal(ax.yaxis.get_label().get_text(), "Customer's Recency")
        plt.close()

    def test_plot_probability_alive_matrix_max_recency(self, bgf):
        shape = (101, 30)
        col_idx = 25
        col = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0.001, 0.002, 0.003, 0.004, 0.006,
               0.008, 0.012, 0.017, 0.023, 0.032, 0.043, 0.058, 0.078, 0.103, 0.134, 0.173,
               0.219, 0.273, 0.333, 0.399, 0.468, 0.537, 0.604, 0.667, 0.724, 0.774, 0.816,
               0.852, 0.882, 0.906, 0.925, 0.941, 0.953, 0.963, 0.970]

        ax = plotting.plot_probability_alive_matrix(bgf, max_recency=100)
        ar = ax.get_images()[0].get_array()
        assert_array_equal(ar.shape, shape)
        assert_allclose(ar[:, col_idx].data, col, atol=0.01)  # only test one column for brevity
        assert_equal(ax.title.get_text(), "Probability Customer is Alive,\nby Frequency and Recency of a Customer")
        assert_equal(ax.xaxis.get_label().get_text(), "Customer's Historical Frequency")
        assert_equal(ax.yaxis.get_label().get_text(), "Customer's Recency")
        plt.close()

    def test_plot_probability_alive_matrix_max_frequency_max_recency(self, bgf):
        shape = (101, 101)
        col_idx = 15
        col = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001,
               0.001, 0.001, 0.002, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.012, 0.016,
               0.020, 0.025, 0.031, 0.039, 0.048, 0.059, 0.072, 0.088, 0.106, 0.126, 0.150,
               0.178, 0.208, 0.242, 0.278, 0.318, 0.359, 0.403, 0.447, 0.492, 0.536, 0.579,
               0.621, 0.660, 0.697, 0.731, 0.763, 0.791, 0.817, 0.839, 0.860, 0.877, 0.893,
               0.907, 0.919, 0.929, 0.939, 0.947, 0.953]

        ax = plotting.plot_probability_alive_matrix(bgf, max_frequency=100, max_recency=100)
        ar = ax.get_images()[0].get_array()
        assert_array_equal(ar.shape, shape)
        assert_allclose(ar[:, col_idx].data, col, atol=0.01)  # only test one column for brevity
        assert_equal(ax.title.get_text(), "Probability Customer is Alive,\nby Frequency and Recency of a Customer")
        assert_equal(ax.xaxis.get_label().get_text(), "Customer's Historical Frequency")
        assert_equal(ax.yaxis.get_label().get_text(), "Customer's Recency")
        plt.close()

    def test_plot_expected_repeat_purchases(self, bgf):
        solid_x_expected = [0.0, 0.39, 0.79, 1.18, 1.57, 1.96, 2.36, 2.75, 3.14, 3.53, 3.93,
                            4.32, 4.71, 5.1, 5.5, 5.89, 6.28, 6.67, 7.07, 7.46, 7.85, 8.24,
                            8.64, 9.03, 9.42, 9.81, 10.21, 10.6, 10.99, 11.38, 11.78, 12.17,
                            12.56, 12.95, 13.35, 13.74, 14.13, 14.52, 14.92, 15.31, 15.7,
                            16.09, 16.49, 16.88, 17.27, 17.66, 18.06, 18.45, 18.84, 19.23,
                            19.63, 20.02, 20.41, 20.8, 21.2, 21.59, 21.98, 22.37, 22.77,
                            23.16, 23.55, 23.94, 24.34, 24.73, 25.12, 25.51, 25.91, 26.3,
                            26.69, 27.08, 27.48, 27.87, 28.26, 28.65, 29.05, 29.44, 29.83,
                            30.22, 30.62, 31.01, 31.4, 31.79, 32.19, 32.58, 32.97, 33.36,
                            33.76, 34.15, 34.54, 34.93, 35.33, 35.72, 36.11, 36.5, 36.9,
                            37.29, 37.68, 38.07, 38.47, 38.86]
        solid_y_expected = [-0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.17, 0.19,
                            0.21, 0.23, 0.24, 0.26, 0.28, 0.29, 0.31, 0.32, 0.34, 0.35, 0.37,
                            0.38, 0.4, 0.41, 0.43, 0.44, 0.45, 0.47, 0.48, 0.49, 0.51, 0.52,
                            0.53, 0.54, 0.56, 0.57, 0.58, 0.59, 0.61, 0.62, 0.63, 0.64, 0.65,
                            0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.76, 0.77, 0.78,
                            0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                            0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.98, 0.99,
                            1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.07, 1.08, 1.09,
                            1.1, 1.11, 1.12, 1.13, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.18, 1.19]
        dashed_x_expected = [38.86, 39.06, 39.25, 39.45, 39.65, 39.84, 40.04, 40.23, 40.43, 40.63,
                             40.82, 41.02, 41.22, 41.41, 41.61, 41.8, 42.0, 42.2, 42.39, 42.59, 42.79,
                             42.98, 43.18, 43.37, 43.57, 43.77, 43.96, 44.16, 44.36, 44.55, 44.75,
                             44.94, 45.14, 45.34, 45.53, 45.73, 45.93, 46.12, 46.32, 46.51, 46.71,
                             46.91, 47.1, 47.3, 47.5, 47.69, 47.89, 48.08, 48.28, 48.48, 48.67,
                             48.87, 49.07, 49.26, 49.46, 49.65, 49.85, 50.05, 50.24, 50.44, 50.64,
                             50.83, 51.03, 51.22, 51.42, 51.62, 51.81, 52.01, 52.21, 52.4, 52.6,
                             52.79, 52.99, 53.19, 53.38, 53.58, 53.78, 53.97, 54.17, 54.36, 54.56,
                             54.76, 54.95, 55.15, 55.35, 55.54, 55.74, 55.93, 56.13, 56.33, 56.52,
                             56.72, 56.92, 57.11, 57.31, 57.5, 57.7, 57.9, 58.09, 58.29]
        dashed_y_expected = [1.19, 1.2, 1.2, 1.2, 1.21, 1.21, 1.22, 1.22, 1.22, 1.23, 1.23, 1.24,
                             1.24, 1.24, 1.25, 1.25, 1.26, 1.26, 1.26, 1.27, 1.27, 1.28, 1.28,
                             1.28, 1.29, 1.29, 1.29, 1.3, 1.3, 1.31, 1.31, 1.31, 1.32, 1.32, 1.32,
                             1.33, 1.33, 1.34, 1.34, 1.34, 1.35, 1.35, 1.35, 1.36, 1.36, 1.37, 1.37,
                             1.37, 1.38, 1.38, 1.38, 1.39, 1.39, 1.39, 1.4, 1.4, 1.41, 1.41, 1.41,
                             1.42, 1.42, 1.42, 1.43, 1.43, 1.43, 1.44, 1.44, 1.44, 1.45, 1.45,
                             1.45, 1.46, 1.46, 1.47, 1.47, 1.47, 1.48, 1.48, 1.48, 1.49, 1.49, 1.49,
                             1.5, 1.5, 1.5, 1.51, 1.51, 1.51, 1.52, 1.52, 1.52, 1.53, 1.53, 1.53,
                             1.54, 1.54, 1.54, 1.55, 1.55, 1.55]

        ax = plotting.plot_expected_repeat_purchases(bgf)
        solid, dashed = ax.lines
        solid_x, solid_y = solid.get_data()
        dashed_x, dashed_y = dashed.get_data()

        # compare the coordinates in the matplotlib axes objects to expected values
        assert_allclose(solid_x, solid_x_expected, atol=0.01)
        assert_allclose(solid_y, solid_y_expected, atol=0.01)
        assert_allclose(dashed_x, dashed_x_expected, atol=0.01)
        assert_allclose(dashed_y, dashed_y_expected, atol=0.01)
        assert_equal(ax.title.get_text(), "Expected Number of Repeat Purchases per Customer")
        assert_equal(ax.xaxis.get_label().get_text(), "Time Since First Purchase")
        assert_equal(ax.yaxis.get_label().get_text(), "")
        plt.close()

    def test_plot_expected_repeat_purchases_with_label(self, bgf):
        solid_x_expected = [0.0, 0.39, 0.79, 1.18, 1.57, 1.96, 2.36, 2.75, 3.14, 3.53, 3.93,
                            4.32, 4.71, 5.1, 5.5, 5.89, 6.28, 6.67, 7.07, 7.46, 7.85, 8.24,
                            8.64, 9.03, 9.42, 9.81, 10.21, 10.6, 10.99, 11.38, 11.78, 12.17,
                            12.56, 12.95, 13.35, 13.74, 14.13, 14.52, 14.92, 15.31, 15.7,
                            16.09, 16.49, 16.88, 17.27, 17.66, 18.06, 18.45, 18.84, 19.23,
                            19.63, 20.02, 20.41, 20.8, 21.2, 21.59, 21.98, 22.37, 22.77,
                            23.16, 23.55, 23.94, 24.34, 24.73, 25.12, 25.51, 25.91, 26.3,
                            26.69, 27.08, 27.48, 27.87, 28.26, 28.65, 29.05, 29.44, 29.83,
                            30.22, 30.62, 31.01, 31.4, 31.79, 32.19, 32.58, 32.97, 33.36,
                            33.76, 34.15, 34.54, 34.93, 35.33, 35.72, 36.11, 36.5, 36.9,
                            37.29, 37.68, 38.07, 38.47, 38.86]
        solid_y_expected = [-0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.17, 0.19,
                            0.21, 0.23, 0.24, 0.26, 0.28, 0.29, 0.31, 0.32, 0.34, 0.35, 0.37,
                            0.38, 0.4, 0.41, 0.43, 0.44, 0.45, 0.47, 0.48, 0.49, 0.51, 0.52,
                            0.53, 0.54, 0.56, 0.57, 0.58, 0.59, 0.61, 0.62, 0.63, 0.64, 0.65,
                            0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.76, 0.77, 0.78,
                            0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                            0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.98, 0.99,
                            1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.07, 1.08, 1.09,
                            1.1, 1.11, 1.12, 1.13, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.18, 1.19]
        dashed_x_expected = [38.86, 39.06, 39.25, 39.45, 39.65, 39.84, 40.04, 40.23, 40.43, 40.63,
                             40.82, 41.02, 41.22, 41.41, 41.61, 41.8, 42.0, 42.2, 42.39, 42.59, 42.79,
                             42.98, 43.18, 43.37, 43.57, 43.77, 43.96, 44.16, 44.36, 44.55, 44.75,
                             44.94, 45.14, 45.34, 45.53, 45.73, 45.93, 46.12, 46.32, 46.51, 46.71,
                             46.91, 47.1, 47.3, 47.5, 47.69, 47.89, 48.08, 48.28, 48.48, 48.67,
                             48.87, 49.07, 49.26, 49.46, 49.65, 49.85, 50.05, 50.24, 50.44, 50.64,
                             50.83, 51.03, 51.22, 51.42, 51.62, 51.81, 52.01, 52.21, 52.4, 52.6,
                             52.79, 52.99, 53.19, 53.38, 53.58, 53.78, 53.97, 54.17, 54.36, 54.56,
                             54.76, 54.95, 55.15, 55.35, 55.54, 55.74, 55.93, 56.13, 56.33, 56.52,
                             56.72, 56.92, 57.11, 57.31, 57.5, 57.7, 57.9, 58.09, 58.29]
        dashed_y_expected = [1.19, 1.2, 1.2, 1.2, 1.21, 1.21, 1.22, 1.22, 1.22, 1.23, 1.23, 1.24,
                             1.24, 1.24, 1.25, 1.25, 1.26, 1.26, 1.26, 1.27, 1.27, 1.28, 1.28,
                             1.28, 1.29, 1.29, 1.29, 1.3, 1.3, 1.31, 1.31, 1.31, 1.32, 1.32, 1.32,
                             1.33, 1.33, 1.34, 1.34, 1.34, 1.35, 1.35, 1.35, 1.36, 1.36, 1.37, 1.37,
                             1.37, 1.38, 1.38, 1.38, 1.39, 1.39, 1.39, 1.4, 1.4, 1.41, 1.41, 1.41,
                             1.42, 1.42, 1.42, 1.43, 1.43, 1.43, 1.44, 1.44, 1.44, 1.45, 1.45,
                             1.45, 1.46, 1.46, 1.47, 1.47, 1.47, 1.48, 1.48, 1.48, 1.49, 1.49, 1.49,
                             1.5, 1.5, 1.5, 1.51, 1.51, 1.51, 1.52, 1.52, 1.52, 1.53, 1.53, 1.53,
                             1.54, 1.54, 1.54, 1.55, 1.55, 1.55]
        label = 'test label'

        ax = plotting.plot_expected_repeat_purchases(bgf, label=label)
        solid, dashed = ax.lines
        legend = plt.gca().legend_
        solid_x, solid_y = solid.get_data()
        dashed_x, dashed_y = dashed.get_data()

        # compare the coordinates in the matplotlib axes objects to expected values
        assert_allclose(solid_x, solid_x_expected, atol=0.01)
        assert_allclose(solid_y, solid_y_expected, atol=0.01)
        assert_allclose(dashed_x, dashed_x_expected, atol=0.01)
        assert_allclose(dashed_y, dashed_y_expected, atol=0.01)
        assert_equal(legend.get_texts()[0].get_text(), label)
        assert_equal(ax.title.get_text(), "Expected Number of Repeat Purchases per Customer")
        assert_equal(ax.xaxis.get_label().get_text(), "Time Since First Purchase")
        assert_equal(ax.yaxis.get_label().get_text(), "")
        plt.close()

    def test_plot_transaction_rate_heterogeneity(self, bgf):
        """Test transactions rate heterogeneity."""

        x_expected = [0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.07,
                      0.07, 0.08, 0.08, 0.09, 0.09, 0.1, 0.1, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14,
                      0.14, 0.15, 0.15, 0.16, 0.16, 0.17, 0.18, 0.18, 0.19, 0.19, 0.2, 0.2, 0.21,
                      0.21, 0.22, 0.23, 0.23, 0.24, 0.24, 0.25, 0.25, 0.26, 0.26, 0.27, 0.27, 0.28,
                      0.29, 0.29, 0.3, 0.3, 0.31, 0.31, 0.32, 0.32, 0.33, 0.34, 0.34, 0.35, 0.35, 0.36,
                      0.36, 0.37, 0.37, 0.38, 0.38, 0.39, 0.4, 0.4, 0.41, 0.41, 0.42, 0.42, 0.43, 0.43,
                      0.44, 0.45, 0.45, 0.46, 0.46, 0.47, 0.47, 0.48, 0.48, 0.49, 0.49, 0.5, 0.51, 0.51,
                      0.52, 0.52, 0.53, 0.53, 0.54, 0.54]
        y_expected = [np.inf, 19.25, 11.11, 7.98, 6.26, 5.16, 4.39, 3.81, 3.36, 3.0, 2.71, 2.46, 2.24,
                      2.06, 1.9, 1.76, 1.64, 1.53, 1.43, 1.34, 1.26, 1.18, 1.11, 1.05, 0.99, 0.94, 0.89,
                      0.84, 0.8, 0.76, 0.72, 0.69, 0.66, 0.63, 0.6, 0.57, 0.55, 0.52, 0.5, 0.48, 0.46,
                      0.44, 0.42, 0.4, 0.39, 0.37, 0.36, 0.34, 0.33, 0.32, 0.3, 0.29, 0.28, 0.27, 0.26,
                      0.25, 0.24, 0.23, 0.22, 0.21, 0.21, 0.2, 0.19, 0.19, 0.18, 0.17, 0.17, 0.16, 0.16,
                      0.15, 0.14, 0.14, 0.13, 0.13, 0.13, 0.12, 0.12, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1,
                      0.09, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06,
                      0.06, 0.06, 0.06]

        ax = plotting.plot_transaction_rate_heterogeneity(bgf)
        x, y = ax.lines[0].get_data()

        assert_allclose(x, x_expected, atol=0.01)
        assert_allclose(y, y_expected, atol=0.01)
        assert_equal(plt.gcf()._suptitle.get_text(), "Heterogeneity in Transaction Rate")
        assert_equal(ax.title.get_text(), "mean: 0.055, var: 0.012")
        assert_equal(ax.xaxis.get_label().get_text(), "Transaction Rate")
        assert_equal(ax.yaxis.get_label().get_text(), "Density")
        plt.close()

    def test_plot_dropout_rate_heterogeneity(self, bgf):
        """Test dropout rate heterogeneity."""

        x_expected = [0.0, 0.01, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.09,
                      0.1, 0.11, 0.12, 0.13, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.18, 0.19,
                      0.2, 0.21, 0.22, 0.23, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.28, 0.29,
                      0.3, 0.31, 0.32, 0.33, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.38, 0.39,
                      0.4, 0.41, 0.42, 0.43, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.48, 0.49,
                      0.5, 0.51, 0.52, 0.53, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.58, 0.59,
                      0.6, 0.61, 0.62, 0.63, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.69,
                      0.7, 0.71, 0.72, 0.73, 0.74, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.79,
                      0.8, 0.81, 0.82, 0.83]
        y_expected = [np.inf, 4.44, 3.8, 3.45, 3.21, 3.03, 2.88, 2.75, 2.65, 2.55, 2.46, 2.38,
                      2.31, 2.24, 2.18, 2.12, 2.06, 2.01, 1.96, 1.91, 1.86, 1.82, 1.77, 1.73,
                      1.69, 1.65, 1.61, 1.58, 1.54, 1.51, 1.47, 1.44, 1.41, 1.37, 1.34, 1.31,
                      1.28, 1.26, 1.23, 1.2, 1.17, 1.14, 1.12, 1.09, 1.07, 1.04, 1.02, 0.99,
                      0.97, 0.95, 0.92, 0.9, 0.88, 0.86, 0.84, 0.81, 0.79, 0.77, 0.75, 0.73,
                      0.71, 0.69, 0.68, 0.66, 0.64, 0.62, 0.6, 0.58, 0.57, 0.55, 0.53, 0.52,
                      0.5, 0.48, 0.47, 0.45, 0.44, 0.42, 0.41, 0.39, 0.38, 0.36, 0.35, 0.33,
                      0.32, 0.31, 0.29, 0.28, 0.27, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.18,
                      0.17, 0.16, 0.15, 0.14]

        ax = plotting.plot_dropout_rate_heterogeneity(bgf)
        x, y = ax.lines[0].get_data()
        print(y)
        assert_allclose(x, x_expected, atol=0.1)
        assert_allclose(y, y_expected, atol=0.1)
        assert_equal(plt.gcf()._suptitle.get_text(), "Heterogeneity in Dropout Probability")
        assert_equal(ax.xaxis.get_label().get_text(), "Dropout Probability p")
        assert_equal(ax.yaxis.get_label().get_text(), "Density")
        plt.close()

    def test_plot_customer_alive_history(self, bgf):
        from datetime import datetime, timedelta

        x_expected = np.arange(datetime(2014, 6, 30), datetime(2015, 1, 17), timedelta(days=1))
        y_expected = [1.0, 1.0, 1.0, 0.75, 0.72, 0.69, 0.67, 0.64, 0.62, 0.59, 0.57, 0.55, 0.81,
                      0.79, 0.77, 0.75, 0.85, 0.87, 0.85, 0.82, 0.8, 0.89, 0.87, 0.84, 0.82, 0.79,
                      0.76, 0.73, 0.69, 0.66, 0.62, 0.59, 0.9, 0.89, 0.87, 0.85, 0.83, 0.81, 0.91,
                      0.9, 0.88, 0.87, 0.85, 0.83, 0.8, 0.78, 0.75, 0.73, 0.7, 0.67, 0.64, 0.61,
                      0.57, 0.54, 0.51, 0.48, 0.45, 0.42, 0.39, 0.37, 0.34, 0.32, 0.29, 0.27, 0.25,
                      0.23, 0.21, 0.2, 0.18, 0.92, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.85,
                      0.84, 0.82, 0.8, 0.79, 0.77, 0.75, 0.73, 0.71, 0.94, 0.93, 0.92, 0.91, 0.9,
                      0.89, 0.88, 0.87, 0.86, 0.85, 0.83, 0.82, 0.8, 0.78, 0.77, 0.94, 0.93, 0.93,
                      0.92, 0.91, 0.9, 0.89, 0.94, 0.94, 0.93, 0.92, 0.92, 0.91, 0.9, 0.89, 0.88,
                      0.87, 0.95, 0.94, 0.94, 0.95, 0.95, 0.94, 0.93, 0.95, 0.95, 0.94, 0.94, 0.93,
                      0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86, 0.96, 0.95, 0.95, 0.94, 0.93, 0.93,
                      0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.85, 0.84, 0.83, 0.81, 0.79, 0.78, 0.76,
                      0.74, 0.72, 0.7, 0.68, 0.66, 0.63, 0.61, 0.59, 0.56, 0.54, 0.52, 0.49, 0.47,
                      0.44, 0.42, 0.4, 0.38, 0.36, 0.34, 0.32, 0.3, 0.28, 0.26, 0.24, 0.23, 0.21,
                      0.2, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.1, 0.09, 0.08,
                      0.08, 0.07, 0.07, 0.06, 0.06, 0.05]
        labels = ['P_alive', 'purchases']

        transaction_data = load_transaction_data()
        # yes I know this is using the wrong data, but I'm testing plotting here.
        id_user = 35
        days_since_birth = 200
        sp_trans = transaction_data.loc[transaction_data['id'] == id_user]
        ax = plotting.plot_history_alive(bgf, days_since_birth, sp_trans, 'date')

        x, y = ax.lines[0].get_data()
        legend = ax.legend_

        assert_allclose([np.round(e, 5) for e in y], y_expected, atol=0.01)  # y has some weird array shapes
        assert_array_equal(x, x_expected)
        assert_array_equal([e.get_text() for e in legend.get_texts()], labels)
        assert_equal(ax.title.get_text(), "History of P_alive")
        assert_equal(ax.xaxis.get_label().get_text(), "")
        assert_equal(ax.yaxis.get_label().get_text(), "P_alive")
        plt.close()

    def test_plot_calibration_purchases_vs_holdout_purchases(self, transaction_data, bgf):
        holdout_expected = [0.161, 0.233, 0.348, 0.544, 0.710, 0.704, 1.606]
        predictions_expected = [0.270, 0.294, 0.402, 0.422, 0.706, 0.809, 1.019]
        labels = ['frequency_holdout', 'model_predictions']

        summary = utils.calibration_and_holdout_data(transaction_data, 'id', 'date', '2014-09-01', '2014-12-31')
        bgf.fit(summary['frequency_cal'], summary['recency_cal'], summary['T_cal'])

        ax = plotting.plot_calibration_purchases_vs_holdout_purchases(bgf, summary)

        lines = ax.lines
        legend = ax.legend_
        holdout = lines[0].get_data()[1]
        predictions = lines[1].get_data()[1]

        assert_allclose(holdout, holdout_expected, atol=0.01)
        assert_allclose(predictions, predictions_expected, atol=0.01)
        assert_array_equal([e.get_text() for e in legend.get_texts()], labels)
        assert_equal(ax.title.get_text(), "Actual Purchases in Holdout Period vs Predicted Purchases")
        assert_equal(ax.xaxis.get_label().get_text(), "Purchases in calibration period")
        assert_equal(ax.yaxis.get_label().get_text(), "Average of Purchases in Holdout Period")
        plt.close()

    def test_plot_calibration_purchases_vs_holdout_purchases_time_since_last_purchase(self, transaction_data, bgf):
        holdout_expected = [3.954, 3.431, 3.482, 3.484, 2.75, 2.289, 1.968]
        predictions_expected = [4.345, 2.993, 3.236, 2.677, 2.240, 2.608, 2.430]
        labels = ['frequency_holdout', 'model_predictions']

        summary = utils.calibration_and_holdout_data(transaction_data, 'id', 'date', '2014-09-01', '2014-12-31')
        bgf.fit(summary['frequency_cal'], summary['recency_cal'], summary['T_cal'])

        ax = plotting.plot_calibration_purchases_vs_holdout_purchases(bgf, summary, kind='time_since_last_purchase')

        lines = ax.lines
        legend = ax.legend_
        holdout = lines[0].get_data()[1]
        predictions = lines[1].get_data()[1]

        assert_allclose(holdout, holdout_expected, atol=0.01)
        assert_allclose(predictions, predictions_expected, atol=0.01)
        assert_array_equal([e.get_text() for e in legend.get_texts()], labels)
        assert_equal(ax.title.get_text(), "Actual Purchases in Holdout Period vs Predicted Purchases")
        assert_equal(ax.xaxis.get_label().get_text(), "Time since user made last purchase")
        assert_equal(ax.yaxis.get_label().get_text(), "Average of Purchases in Holdout Period")
        plt.close()

    def test_plot_cumulative_transactions(self, cdnow_transactions, bgf_transactions):
        """Test plotting cumultative transactions with CDNOW example."""

        actual = [0, 3, 17, 44, 67, 122, 173, 240, 313, 375, 466,
                  555, 655, 739, 825, 901, 970, 1033, 1091, 1159, 1217, 1277,
                  1325, 1367, 1444, 1528, 1584, 1632, 1675, 1741, 1813, 1846, 1894,
                  1954, 2002, 2051, 2094, 2141, 2195, 2248, 2299, 2344, 2401, 2452,
                  2523, 2582, 2636, 2685, 2739, 2805, 2860, 2891, 2933, 2983, 3023,
                  3057, 3099, 3140, 3184, 3226, 3283, 3344, 3400, 3456, 3517, 3553,
                  3592, 3632, 3661, 3699, 3740, 3770, 3802, 3842, 3887, 3939, 3967,
                  4001]
        predicted = [4.089e+00, 1.488e+01, 3.240e+01, 5.716e+01, 8.939e+01, 1.297e+02,
                     1.769e+02, 2.310e+02, 2.927e+02, 3.616e+02, 4.369e+02, 5.174e+02,
                     5.984e+02, 6.775e+02, 7.549e+02, 8.307e+02, 9.052e+02, 9.784e+02,
                     1.050e+03, 1.121e+03, 1.191e+03, 1.260e+03, 1.328e+03, 1.396e+03,
                     1.462e+03, 1.528e+03, 1.594e+03, 1.658e+03, 1.722e+03, 1.786e+03,
                     1.849e+03, 1.911e+03, 1.973e+03, 2.035e+03, 2.096e+03, 2.156e+03,
                     2.216e+03, 2.276e+03, 2.335e+03, 2.394e+03, 2.452e+03, 2.511e+03,
                     2.568e+03, 2.626e+03, 2.683e+03, 2.740e+03, 2.797e+03, 2.853e+03,
                     2.909e+03, 2.964e+03, 3.020e+03, 3.075e+03, 3.130e+03, 3.185e+03,
                     3.239e+03, 3.293e+03, 3.347e+03, 3.401e+03, 3.454e+03, 3.507e+03,
                     3.560e+03, 3.613e+03, 3.666e+03, 3.718e+03, 3.771e+03, 3.823e+03,
                     3.874e+03, 3.926e+03, 3.978e+03, 4.029e+03, 4.080e+03, 4.131e+03,
                     4.182e+03, 4.232e+03, 4.283e+03, 4.333e+03, 4.383e+03, 4.433e+03]
        labels = ['actual', 'predicted']
        t = 39
        freq = 'W'

        ax = plotting.plot_cumulative_transactions(
            bgf_transactions, cdnow_transactions, 'date', 'id_sample', 2 * t,
            t, freq=freq, xlabel='week', datetime_format='%Y%m%d')

        lines = ax.lines
        legend = ax.legend_

        actual_y = lines[0].get_data()[1]
        predicted_y = lines[1].get_data()[1]
        assert_allclose(actual, actual_y, rtol=0.01)
        assert_allclose(predicted, predicted_y, rtol=0.01)
        assert_array_equal([e.get_text() for e in legend.get_texts()], labels)
        assert_equal(ax.title.get_text(), "Tracking Cumulative Transactions")
        assert_equal(ax.xaxis.get_label().get_text(), "week")
        assert_equal(ax.yaxis.get_label().get_text(), "Cumulative Transactions")
        plt.close()

    def test_plot_incremental_transactions(self, cdnow_transactions, bgf_transactions):
        """Test plotting incremental transactions with CDNOW example."""
        actual = [np.nan, 3.0, 14.0, 27.0, 23.0, 55.0, 51.0, 67.0, 73.0, 62.0, 91.0, 89.0, 100.0,
                  84.0, 86.0, 76.0, 69.0, 63.0, 58.0, 68.0, 58.0, 60.0, 48.0, 42.0, 77.0, 84.0,
                  56.0, 48.0, 43.0, 66.0, 72.0, 33.0, 48.0, 60.0, 48.0, 49.0, 43.0, 47.0, 54.0,
                  53.0, 51.0, 45.0, 57.0, 51.0, 71.0, 59.0, 54.0, 49.0, 54.0, 66.0, 55.0, 31.0,
                  42.0, 50.0, 40.0, 34.0, 42.0, 41.0, 44.0, 42.0, 57.0, 61.0, 56.0, 56.0, 61.0,
                  36.0, 39.0, 40.0, 29.0, 38.0, 41.0, 30.0, 32.0, 40.0, 45.0, 52.0, 28.0, 34.0]
        predicted = [np.nan, 10.79, 17.52, 24.76, 32.23, 40.38, 47.16, 54.12, 61.71, 68.88, 75.35,
                     80.49, 81.0, 79.08, 77.38, 75.85, 74.47, 73.21, 72.06, 71.0, 70.01, 69.09, 68.24,
                     67.43, 66.68, 65.97, 65.29, 64.66, 64.05, 63.47, 62.92, 62.4, 61.9, 61.41, 60.95,
                     60.51, 60.08, 59.67, 59.28, 58.9, 58.53, 58.17, 57.82, 57.49, 57.17, 56.85, 56.55,
                     56.25, 55.96, 55.68, 55.41, 55.15, 54.89, 54.64, 54.39, 54.15, 53.92, 53.69, 53.47,
                     53.25, 53.04, 52.83, 52.62, 52.42, 52.23, 52.04, 51.85, 51.67, 51.48, 51.31, 51.13,
                     50.96, 50.8, 50.63, 50.47, 50.31, 50.16, 50.0]
        labels = ['actual', 'predicted']
        t = 39
        freq = 'W'

        ax = plotting.plot_incremental_transactions(
            bgf_transactions, cdnow_transactions, 'date', 'id_sample', 2 * t, t, freq=freq,
            xlabel='week', datetime_format='%Y%m%d')

        lines = ax.lines
        legend = ax.legend_

        actual_y = lines[0].get_data()[1]
        predicted_y = lines[1].get_data()[1]

        assert_allclose(actual, actual_y, rtol=0.01)
        assert_allclose(predicted, predicted_y, rtol=0.01)
        assert_array_equal([e.get_text() for e in legend.get_texts()], labels)
        assert_equal(ax.title.get_text(), "Tracking Daily Transactions")
        assert_equal(ax.xaxis.get_label().get_text(), "week")
        assert_equal(ax.yaxis.get_label().get_text(), "Transactions")
        plt.close()
