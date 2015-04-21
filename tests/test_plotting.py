import os
import pandas as pd
import pytest

from lifetimes import plotting
from lifetimes import BetaGeoFitter


bgf = BetaGeoFitter()
data = pd.read_csv('lifetimes/datasets/cdnow_customers.csv', index_col=[0])
bgf.fit(data['frequency'], data['recency'], data['T'], iterative_fitting=0)

@pytest.mark.plottest
@pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
class TestPlotting():
    
    def test_plot_period_transactions(self):
        from matplotlib import pyplot as plt
        
        plt.figure()
        plotting.plot_period_transactions(bgf)
        
        plt.figure()
        plotting.plot_period_transactions(bgf, bins=range(5))
        
        plt.figure()
        plotting.plot_period_transactions(bgf, label=['A', 'B'])
        plt.show()

    def test_plot_frequency_recency_matrix(self):
        from matplotlib import pyplot as plt

        plt.figure()
        plotting.plot_frequency_recency_matrix(bgf)

        plt.figure()
        plotting.plot_frequency_recency_matrix(bgf, max_recency=100, max_frequency=50)

        plt.show()

    def test_plot_expected_repeat_purchases(self):
        from matplotlib import pyplot as plt

        plt.figure()
        plotting.plot_expected_repeat_purchases(bgf)

        plt.figure()
        plotting.plot_expected_repeat_purchases(bgf, label='test label')

        plt.show()



