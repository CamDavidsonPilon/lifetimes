import os
import pandas as pd
import pytest

from lifetimes import plotting
from lifetimes import BetaGeoFitter


BG = BetaGeoFitter()
data = pd.read_csv('lifetimes/datasets/cdnow_customers.csv', index_col=[0])
BG.fit(data['x'], data['t_x'], data['T'], iterative_fitting=0)

@pytest.mark.plottest
@pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
class TestPlotting():
    
    def test_plot_period_transactions(self):
        from matplotlib import pyplot as plt
        
        plt.figure()
        plotting.plot_period_transactions(BG)
        
        plt.figure()
        plotting.plot_period_transactions(BG, bins=range(5))
        
        plt.figure()
        plotting.plot_period_transactions(BG, label=['A', 'B'])
        plt.show()

    def test_plot_frequency_recency_matrix(self):
        from matplotlib import pyplot as plt

        plt.figure()
        plotting.plot_frequency_recency_matrix(BG)

        plt.figure()
        plotting.plot_frequency_recency_matrix(BG, max_t=100, max_x=50)

        plt.show()

    def test_plot_expected_repeat_purchases(self):
        from matplotlib import pyplot as plt

        plt.figure()
        plotting.plot_expected_repeat_purchases(BG)

        plt.figure()
        plotting.plot_expected_repeat_purchases(BG, label='test label')

        plt.show()



