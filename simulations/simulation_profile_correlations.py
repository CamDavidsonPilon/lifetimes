from __future__ import print_function
import pytest
import math
import lifetimes.generate_data as gen
import numpy as np
import lifetimes.models as mod
from lifetimes import models
from lifetimes.data_compression import compress_data, compress_session_session_before_conversion_data
import pandas as pd
from lifetimes.estimation import BGBBFitter, BGBBBGExtFitter
from uncertainties import ufloat, UFloat, correlation_matrix
import matplotlib.pyplot as plt


@pytest.mark.sim_correlation
def test_BGBB_correlations():
    T = 10
    size = 100
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7}

    data = gen.bgbb_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'], size=size)

    data = compress_data(data)

    model = models.BGBBModel()

    model.fit(data['frequency'], data['recency'], data['T'], bootstrap_size=10, N=data['N'],
              initial_params=params.values())

    print("Generation params")
    print(params)

    print("Fitted params")
    print(model.params)
    print(model.params_C)

    ts = range(1, 50)
    cum_profile_points = [model.expected_number_of_purchases_up_to_time(t) for t in [0] + ts]
    diff_profile_points = [cum_profile_points[t] - cum_profile_points[t-1] for t in ts]

    cor_matrix = correlation_matrix(diff_profile_points)

    fig, ax1 = plt.subplots()
    t = np.arange(0.01, 10.0, 0.01)
    s1 = np.exp(t)
    ax1.errorbar(ts, y=[p.n for p in diff_profile_points], yerr=[p.s for p in diff_profile_points], fmt='o')
    ax1.set_xlabel('time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('quantity', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    for b in [0, 1, 10, 20]:
        plt.plot(cor_matrix[b], label='bin ' + str(b))
    ax2.set_ylabel('correlation', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.legend()
    plt.show()



@pytest.mark.sim_correlation
def test_BGBBBG_correlations():
    T = 10
    size = 100

    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0, 'c0': 0.05}

    data = gen.bgbbbgext_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                               params['epsilon'],
                               params['zeta'], params['c0'], size=size, time_first_purchase=True)

    compressed_data = compress_session_session_before_conversion_data(data)

    model = mod.BGBBBGExtModel(penalizer_coef=0.2)

    model.fit(frequency=compressed_data['frequency'], recency=compressed_data['recency'], T=compressed_data['T'],
              frequency_before_conversion=compressed_data['frequency_before_conversion'],
              N=compressed_data['N'], initial_params=params.values())

    print("Generation params")
    print(params)

    print("Fitted params")
    print(model.params)
    print(model.params_C)

    print("Uncertain parameters")
    print(model.uparams)

    ts = range(0, 50)
    diff_profile_points = [model.expected_probability_of_converting_at_time(t) for t in ts]

    cor_matrix = correlation_matrix(diff_profile_points)

    fig, ax1 = plt.subplots()
    t = np.arange(0.01, 10.0, 0.01)
    s1 = np.exp(t)
    ax1.errorbar(ts, y=[p.n for p in diff_profile_points], yerr=[p.s for p in diff_profile_points], fmt='o')
    ax1.set_xlabel('time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('quantity', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    for b in [0, 1, 10, 20]:
        plt.plot(cor_matrix[b], label='bin ' + str(b))
    ax2.set_ylabel('correlation', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.legend()
    plt.show()
