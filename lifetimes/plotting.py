import numpy as np
from lifetimes.utils import coalesce

__all__ = [
    'plot_period_transactions',
    'plot_calibration_purchases_vs_holdout_purchases',
    'plot_frequency_recency_matrix',
    'plot_expected_repeat_purchases',
]


def plot_period_transactions(model, **kwargs):
    from matplotlib import pyplot as plt

    bins = kwargs.pop('bins', range(9))
    labels = kwargs.pop('label', ['Actual', 'Model'])

    n = model.data.shape[0]
    simulated_data = model.generate_new_data(size=n)

    ax = plt.hist(np.c_[model.data['frequency'].values, simulated_data['frequency'].values],
                  bins=bins, label=labels)
    plt.legend()
    plt.xticks(np.arange(len(bins))[:-1] + 0.5, bins[:-1])
    plt.title('Frequency of Repeat Transactions')
    plt.ylabel('Customers')
    plt.xlabel('Number of Calibration Period Transactions')
    return ax


def plot_calibration_purchases_vs_holdout_purchases(model, calibration_holdout_matrix, n=7):
    """
    This currently relies too much on the lifetimes.util calibration_and_holdout_data function.

    """
    from matplotlib import pyplot as plt

    summary = calibration_holdout_matrix.copy()
    T = summary.iloc[0]['duration_holdout']

    summary['model'] = summary.apply(lambda r: model.conditional_expected_number_of_purchases_up_to_time(T, r['frequency_cal'], r['recency_cal'], r['T_cal']), axis=1)

    ax = summary.groupby('frequency_cal')[['frequency_holdout', 'model']].mean().ix[:n].plot()

    plt.title('Actual Purchases in Holdout Period vs Predicted Purchases')
    plt.xlabel('Puchases in Calibration Period')
    plt.ylabel('Average of Purchases in Holdout Period')
    plt.legend()

    return ax


def plot_frequency_recency_matrix(model, max_x=None, max_t=None, **kwargs):
    from matplotlib import pyplot as plt

    if max_x is None:
        max_x = int(model.data['frequency'].max())

    if max_t is None:
        max_t = int(model.data['T'].max())

    t = 1  # one unit of time
    Z = np.zeros((max_t, max_x))
    for i, t_x in enumerate(np.arange(max_t)):
        for j, x in enumerate(np.arange(max_x)):
            Z[i, j] = model.conditional_expected_number_of_purchases_up_to_time(t, x, t_x, max_t)

    interpolation = kwargs.pop('interpolation', 'none')

    ax = plt.subplot(111)
    ax.imshow(Z, interpolation=interpolation, **kwargs)
    plt.xlabel("Customer's Historical Frequency")
    plt.ylabel("Customer's Recency")
    plt.title('Expected Number of Future Purchases over 1 Unit of Time,\nby Frequency and Recency of a Customer')

    # turn matrix into square
    forceAspect(ax)

    # necessary for colorbar to show up
    PCM = ax.get_children()[2]
    plt.colorbar(PCM, ax=ax)

    return ax


def plot_expected_repeat_purchases(model, **kwargs):
    from matplotlib import pyplot as plt

    ax = kwargs.pop('ax', None) or plt.subplot(111)
    color_cycle = ax._get_lines.color_cycle

    label = kwargs.pop('label', None)
    color = coalesce(kwargs.pop('c', None), kwargs.pop('color', None), next(color_cycle))
    max_T = model.data['T'].max()

    times = np.linspace(0, max_T, 100)
    ax = plt.plot(times, model.expected_number_of_purchases_up_to_time(times), color=color, label=label, **kwargs)

    times = np.linspace(max_T, 1.5 * max_T, 100)
    plt.plot(times, model.expected_number_of_purchases_up_to_time(times), color=color, ls='--', **kwargs)

    plt.title('Expected Number of Repeat Purchases per Customer')
    plt.xlabel('Time Since First Purchase')
    plt.legend(loc='lower right')
    return ax


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
