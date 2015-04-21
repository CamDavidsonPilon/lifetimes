import os
import pandas as pd
import pytest

from lifetimes import plotting
from lifetimes import BetaGeoFitter


bfg = BetaGeoFitter()
data = pd.read_csv('lifetimes/datasets/cdnow_customers.csv', index_col=[0])
bfg.fit(data['frequency'], data['recency'], data['T'], iterative_fitting=0)

@pytest.mark.plottest
@pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
class TestPlotting():
    
    def test_plot_period_transactions(self):
        from matplotlib import pyplot as plt
        
        plt.figure()
        plotting.plot_period_transactions(bfg)
        
        plt.figure()
        plotting.plot_period_transactions(bfg, bins=range(5))
        
        plt.figure()
        plotting.plot_period_transactions(bfg, label=['A', 'B'])
        plt.show()

    def test_plot_frequency_recency_matrix(self):
        from matplotlib import pyplot as plt

        plt.figure()
        plotting.plot_frequency_recency_matrix(bfg)

        plt.figure()
        plotting.plot_frequency_recency_matrix(bfg, max_recency=100, max_frequency=50)

        plt.show()

    def test_plot_expected_repeat_purchases(self):
        from matplotlib import pyplot as plt

        plt.figure()
        plotting.plot_expected_repeat_purchases(bfg)

        plt.figure()
        plotting.plot_expected_repeat_purchases(bfg, label='test label')

        plt.show()



