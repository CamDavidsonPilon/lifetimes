import numpy as np
import pandas as pd
from lifetimes.utils import coalesce, calculate_alive_path, expected_cumulative_transactions
from scipy import stats

__all__ = [
    'plot_period_transactions',
    'plot_calibration_purchases_vs_holdout_purchases',
    'plot_frequency_recency_matrix',
    'plot_probability_alive_matrix',
    'plot_expected_repeat_purchases',
    'plot_history_alive',
    'plot_cumulative_transactions',
    'plot_incremental_transactions',
    'plot_transaction_rate_heterogeneity',
    'plot_dropout_rate_heterogeneity'
]


def plot_period_transactions(model, max_frequency=7, title='Frequency of Repeat Transactions',
                             xlabel='Number of Calibration Period Transactions', ylabel='Customers', **kwargs):
    """
    Plot a figure with period actual and predicted transactions

    Parameters:
        model: a fitted lifetimes model.
        max_frequency: the maximum frequency to plot.
        title: figure title
        xlabel: figure xlabel
        ylabel: figure ylabel
        kwargs: passed into the matplotlib.pyplot.plot command.
    """
    from matplotlib import pyplot as plt
    labels = kwargs.pop('label', ['Actual', 'Model'])

    n = model.data.shape[0]
    simulated_data = model.generate_new_data(size=n)

    model_counts = pd.DataFrame(model.data['frequency'].value_counts().sort_index().iloc[:max_frequency])
    simulated_counts = pd.DataFrame(simulated_data['frequency'].value_counts().sort_index().iloc[:max_frequency])
    combined_counts = model_counts.merge(simulated_counts, how='outer', left_index=True, right_index=True).fillna(0)
    combined_counts.columns = labels

    ax = combined_counts.plot(kind='bar', **kwargs)

    plt.legend()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return ax


def plot_calibration_purchases_vs_holdout_purchases(model, calibration_holdout_matrix, kind="frequency_cal", n=7, **kwargs):
    """
    This currently relies too much on the lifetimes.util calibration_and_holdout_data function.

    Parameters:
        model: a fitted lifetimes model
        calibration_holdout_matrix: dataframe from calibration_and_holdout_data function
        kind: x-axis :"frequency_cal". Purchases in calibration period,
                      "recency_cal". Age of customer at last purchase,
                      "T_cal". Age of customer at the end of calibration period,
                      "time_since_last_purchase". Time since user made last purchase
        n: number of ticks on the x axis

    """
    from matplotlib import pyplot as plt

    x_labels = {
        "frequency_cal": "Purchases in calibration period",
        "recency_cal": "Age of customer at last purchase",
        "T_cal": "Age of customer at the end of calibration period",
        "time_since_last_purchase": "Time since user made last purchase"
    }
    summary = calibration_holdout_matrix.copy()
    duration_holdout = summary.iloc[0]['duration_holdout']

    summary['model_predictions'] = summary.apply(lambda r: model.conditional_expected_number_of_purchases_up_to_time(duration_holdout, r['frequency_cal'], r['recency_cal'], r['T_cal']), axis=1)

    if kind == "time_since_last_purchase":
        summary["time_since_last_purchase"] = summary["T_cal"] - summary["recency_cal"]
        ax = summary.groupby(["time_since_last_purchase"])[['frequency_holdout', 'model_predictions']].mean().iloc[:n].plot(**kwargs)
    else:
        ax = summary.groupby(kind)[['frequency_holdout', 'model_predictions']].mean().iloc[:n].plot(**kwargs)

    plt.title('Actual Purchases in Holdout Period vs Predicted Purchases')
    plt.xlabel(x_labels[kind])
    plt.ylabel('Average of Purchases in Holdout Period')
    plt.legend()

    return ax


def plot_frequency_recency_matrix(model, T=1, max_frequency=None, max_recency=None, **kwargs):
    """
    Plot a figure of expected transactions in T next units of time by a customer's
    frequency and recency.

    Parameters:
        model: a fitted lifetimes model.
        T: next units of time to make predictions for
        max_frequency: the maximum frequency to plot. Default is max observed frequency.
        max_recency: the maximum recency to plot. This also determines the age of the customer.
            Default to max observed age.
        kwargs: passed into the matplotlib.imshow command.

    """
    from matplotlib import pyplot as plt

    if max_frequency is None:
        max_frequency = int(model.data['frequency'].max())

    if max_recency is None:
        max_recency = int(model.data['T'].max())

    Z = np.zeros((max_recency + 1, max_frequency + 1))
    for i, recency in enumerate(np.arange(max_recency + 1)):
        for j, frequency in enumerate(np.arange(max_frequency + 1)):
            Z[i, j] = model.conditional_expected_number_of_purchases_up_to_time(T, frequency, recency, max_recency)

    interpolation = kwargs.pop('interpolation', 'none')

    ax = plt.subplot(111)
    PCM = ax.imshow(Z, interpolation=interpolation, **kwargs)
    plt.xlabel("Customer's Historical Frequency")
    plt.ylabel("Customer's Recency")
    plt.title('Expected Number of Future Purchases for %d Unit%s of Time,'
              '\nby Frequency and Recency of a Customer' % (T, "s"[T == 1:]))

    # turn matrix into square
    forceAspect(ax)

    # plot colorbar beside matrix
    plt.colorbar(PCM, ax=ax)

    return ax


def plot_probability_alive_matrix(model, max_frequency=None, max_recency=None,
                                  title='Probability Customer is Alive,\nby Frequency and Recency of a Customer',
                                  xlabel="Customer's Historical Frequency", ylabel="Customer's Recency",
                                  **kwargs):
    """
    Plot a figure of the probability a customer is alive based on their
    frequency and recency.

    Parameters:
        model: a fitted lifetimes model.
        max_frequency: the maximum frequency to plot. Default is max observed frequency.
        max_recency: the maximum recency to plot. This also determines the age of the customer.
            Default to max observed age.
        title: figure title
        xlabel: figure xlabel
        ylabel: figure ylabel
        kwargs: passed into the matplotlib.imshow command.
    """
    from matplotlib import pyplot as plt

    z = model.conditional_probability_alive_matrix(max_frequency, max_recency)

    interpolation = kwargs.pop('interpolation', 'none')

    ax = plt.subplot(111)
    PCM = ax.imshow(z, interpolation=interpolation, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # turn matrix into square
    forceAspect(ax)

    # plot colorbar beside matrix
    plt.colorbar(PCM, ax=ax)

    return ax


def plot_expected_repeat_purchases(model, title='Expected Number of Repeat Purchases per Customer',
                                   xlabel='Time Since First Purchase', **kwargs):
    """
    Plot expected repeat purchases on calibration period .

    Parameters:
        model: a fitted lifetimes model.
        title: figure title
        xlabel: figure xlabel
        ylabel: figure ylabel
        kwargs: passed into the matplotlib.pyplot.plot command.
    """
    from matplotlib import pyplot as plt

    ax = kwargs.pop('ax', None) or plt.subplot(111)
    label = kwargs.pop('label', None)

    if plt.matplotlib.__version__ >= "1.5":
        color_cycle = ax._get_lines.prop_cycler
        color = coalesce(kwargs.pop('c', None), kwargs.pop('color', None), next(color_cycle)['color'])
    else:
        color_cycle = ax._get_lines.color_cycle
        color = coalesce(kwargs.pop('c', None), kwargs.pop('color', None), next(color_cycle))

    max_T = model.data['T'].max()

    times = np.linspace(0, max_T, 100)
    ax = plt.plot(times, model.expected_number_of_purchases_up_to_time(times), color=color, label=label, **kwargs)

    times = np.linspace(max_T, 1.5 * max_T, 100)
    plt.plot(times, model.expected_number_of_purchases_up_to_time(times), color=color, ls='--', **kwargs)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend(loc='lower right')
    return ax


def plot_history_alive(model, t, transactions, datetime_col, freq='D', **kwargs):
    """
    Draws a graph showing the probablility of being alive for a customer in time
    :param model: A fitted lifetimes model
    :param t: the number of time units since the birth we want to draw the p_alive
    :param transactions: a Pandas DataFrame containing the transactions history of the customer_id
    :param datetime_col: the column in the transactions that denotes the datetime the purchase was made
    :param freq: Default 'D' for days. Other examples= 'W' for weekly
    """

    from matplotlib import pyplot as plt

    start_date = kwargs.pop('start_date', min(transactions[datetime_col]))
    ax = kwargs.pop('ax', None) or plt.subplot(111)

    # Get purchasing history of user
    customer_history = transactions[[datetime_col]].copy()
    customer_history.index = pd.DatetimeIndex(customer_history[datetime_col])

    # Add transactions column
    customer_history['transactions'] = 1
    customer_history = customer_history.resample(freq).sum()

    # plot alive_path
    path = calculate_alive_path(model, transactions, datetime_col, t, freq)
    path_dates = pd.date_range(start=min(transactions[datetime_col]), periods=len(path), freq=freq)
    plt.plot(path_dates, path, '-', label='P_alive')

    # plot buying dates
    payment_dates = customer_history[customer_history['transactions'] >= 1].index
    plt.vlines(payment_dates.values, ymin=0, ymax=1, colors='r', linestyles='dashed', label='purchases')

    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlim(start_date, path_dates[-1])
    plt.legend(loc=3)
    plt.ylabel('P_alive')
    plt.title('History of P_alive')

    return ax


def plot_cumulative_transactions(model, transactions, datetime_col, customer_id_col, t, t_cal,
                                 datetime_format=None, freq='D', set_index_date=False,
                                 title='Tracking Cumulative Transactions',
                                 xlabel='day', ylabel='Cumulative Transactions',
                                 **kwargs):
    """
    Plot a figure of the predicted and actual cumulative transactions of users
    Parameters:
        model: A fitted lifetimes model
        transactions: a Pandas DataFrame containing the transactions history of the customer_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        customer_id_col: the column in transactions that denotes the customer_id
        t: the number of time units since the begining of
            data for which we want to calculate cumulative transactions
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        freq: Default 'D' for days, 'W' for weeks, 'M' for months... etc. Full list here:
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
        set_index_date: when True set date as Pandas DataFrame index, default False - number of time units
        title: figure title
        xlabel: figure xlabel, if set_index_date is True will be overwrited to date
        ylabel: figure ylabel
        kwargs: passed into the pandas.DataFrame.plot command.
    """
    from matplotlib import pyplot as plt

    ax = kwargs.pop('ax', None) or plt.subplot(111)

    df_cum_transactions = expected_cumulative_transactions(model, transactions, datetime_col,
                                                           customer_id_col, t,
                                                           datetime_format=datetime_format, freq=freq,
                                                           set_index_date=set_index_date)

    ax = df_cum_transactions.plot(ax=ax, title=title, **kwargs)

    if set_index_date:
        x_vline = df_cum_transactions.index[int(t_cal)]
        xlabel = 'date'
    else:
        x_vline = t_cal
    ax.axvline(x=x_vline, color='r', linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_incremental_transactions(model, transactions, datetime_col, customer_id_col, t, t_cal,
                                  datetime_format=None, freq='D', set_index_date=False,
                                  title='Tracking Daily Transactions',
                                  xlabel='day', ylabel='Transactions',
                                  **kwargs):
    """
    Plot a figure of the predicted and actual cumulative transactions of users
    Parameters:
        model: A fitted lifetimes model
        transactions: a Pandas DataFrame containing the transactions history of the customer_id
        datetime_col: the column in transactions that denotes the datetime the purchase was made.
        customer_id_col: the column in transactions that denotes the customer_id
        t: the number of time units since the begining of
            data for which we want to calculate cumulative transactions
        datetime_format: a string that represents the timestamp format. Useful if Pandas can't understand
            the provided format.
        freq: Default 'D' for days, 'W' for weeks, 'M' for months... etc. Full list here:
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
        set_index_date: when True set date as Pandas DataFrame index, default False - number of time units
        title: figure title
        xlabel: figure xlabel, if set_index_date is True will be overwrited to date
        ylabel: figure ylabel
        kwargs: passed into the pandas.DataFrame.plot command.
    """
    from matplotlib import pyplot as plt

    ax = kwargs.pop('ax', None) or plt.subplot(111)

    df_cum_transactions = expected_cumulative_transactions(model, transactions, datetime_col,
                                                           customer_id_col, t,
                                                           datetime_format=datetime_format, freq=freq,
                                                           set_index_date=set_index_date)

    # get incremental from cumulative transactions
    df_cum_transactions = df_cum_transactions.apply(lambda x: x - x.shift(1))
    ax = df_cum_transactions.plot(ax=ax, title=title, **kwargs)

    if set_index_date:
        x_vline = df_cum_transactions.index[int(t_cal)]
        xlabel = 'date'
    else:
        x_vline = t_cal
    ax.axvline(x=x_vline, color='r', linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_transaction_rate_heterogeneity(model, suptitle='Heterogeneity in Transaction Rate',
                                        xlabel='Transaction Rate', ylabel='Density', **kwargs):
    """
    Plot the estimated gamma distribution of lambda (customers' propensities to purchase).

    Parameters:
        model: A fitted lifetimes model, for now only for BG/NBD
        suptitle: figure suptitle
        xlabel: figure xlabel
        ylabel: figure ylabel
        kwargs: passed into the matplotlib.pyplot.plot command.
    """
    from matplotlib import pyplot as plt

    r, alpha = model._unload_params('r', 'alpha')
    rate_mean = r / alpha
    rate_var = r / alpha**2

    rv = stats.gamma(r, scale=1 / alpha)
    lim = rv.ppf(0.99)
    x = np.linspace(0, lim, 100)

    fig, ax = plt.subplots(1)
    fig.subplots_adjust(top=0.93)
    fig.suptitle('Heterogeneity in Transaction Rate', fontsize=14, fontweight='bold')

    ax.set_title('mean: {:.3f}, var: {:.3f}'.format(rate_mean, rate_var))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.plot(x, rv.pdf(x))
    return ax


def plot_dropout_rate_heterogeneity(model, suptitle='Heterogeneity in Dropout Probability',
                                    xlabel='Dropout Probability p', ylabel='Density', **kwargs):
    """
    Plot the estimated gamma distribution of p (customers' probability of dropping out immediately after a transaction).
    Parameters:
        model: A fitted lifetimes model, for now only for BG/NBD
        suptitle: figure suptitle
        xlabel: figure xlabel
        ylabel: figure ylabel
        kwargs: passed into the matplotlib.pyplot.plot command.
    """
    from matplotlib import pyplot as plt

    a, b = model._unload_params('a', 'b')
    beta_mean = a / (a + b)
    beta_var = a * b / ((a + b)**2) / (a + b + 1)

    rv = stats.beta(a, b)
    lim = rv.ppf(0.99)
    x = np.linspace(0, lim, 100)

    fig, ax = plt.subplots(1)
    fig.subplots_adjust(top=0.93)
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')

    ax.set_title('mean: {:.3f}, var: {:.3f}'.format(beta_mean, beta_var))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.plot(x, rv.pdf(x), **kwargs)
    return ax


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
