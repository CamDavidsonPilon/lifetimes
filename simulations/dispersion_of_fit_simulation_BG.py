import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lifetimes.generate_data as gen
import lifetimes.data_compression as comp
from lifetimes.models import BGModel


def get_estimates(params, probs, daily_installs, observed_days, conversion_rate, free_trial_conversion, N):
    Ts = reduce(lambda x, y: x + y,
                [[(observed_days - day) / 7] * int(math.floor(installs * conversion_rate * free_trial_conversion))
                 for day, installs in enumerate(daily_installs)])
    Ts = filter(lambda x: x > 0, Ts)
    exs = []
    for i in range(N):
        gen_data = gen.bgext_model(Ts, params['alpha'], params['beta'], probs=probs)
        data = comp.compress_bgext_data(gen_data)

        model = BGModel(penalizer_coef=0.1)

        model.fit(data['frequency'], data['T'], bootstrap_size=30, N=data['N'])

        Ex = model.expected_number_of_purchases_up_to_time(52) + 1
        print (i, Ex)
        exs.append(Ex)
    return exs


def set_plot_title(true_Ex, N, daily_installs, conversion_rate, free_trial_conversion):
    plt.title(
        'Histogram of '
        + str(N)
        + ' estimates (true value in red) - ' + str(true_Ex)
        + ' - '
        + str(np.mean(daily_installs) * conversion_rate * free_trial_conversion)
        + ' conv/day')
    plt.xlabel('estimates')
    plt.axvline(x=true_Ex, color="red")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_estimates_lines(exs, true_Ex, number_of_days):
    percentage_out_lower = len(filter(lambda x: true_Ex > x + x.s, exs)) * 100.0 / len(exs)
    percentage_out_upper = len(filter(lambda x: true_Ex < x - x.s, exs)) * 100.0 / len(exs)
    percentage_in = len(filter(lambda x: x - x.s < true_Ex < x + x.s, exs)) * 100.0 / len(exs)
    estimates = map(lambda x: x.n, exs)
    hist = np.histogram(a=map(lambda x: x.n, exs), bins=range(0, 40, 2))
    plt.plot(hist[1][:-1], hist[0], label='Days:{}, Val: {}+/{}, {}%-{}%-{}%'.format(
        number_of_days,
        round(np.mean(estimates), 1),
        round(np.std(estimates), 1),
        round(percentage_out_lower, 1),
        round(percentage_in, 1),
        round(percentage_out_upper, 1)
    ))


def print_estimates_hist(exs, true_Ex, number_of_days, color, edge_color):
    percentage_out_lower = len(filter(lambda x: true_Ex > x + x.s, exs)) * 100.0 / len(exs)
    percentage_out_upper = len(filter(lambda x: true_Ex < x - x.s, exs)) * 100.0 / len(exs)
    percentage_in = len(filter(lambda x: x - x.s < true_Ex < x + x.s, exs)) * 100.0 / len(exs)
    estimates = map(lambda x: x.n, exs)
    plt.hist(
        map(lambda x: x.n, exs),
        bins=range(40), normed=0,
        label='Days:{}, {}+/{}, {}%-{}%-{}%'.format(
            number_of_days,
            round(np.mean(estimates), 1),
            round(np.std(estimates), 1),
            round(percentage_out_lower, 1),
            round(percentage_in, 1),
            round(percentage_out_upper, 1)
        ),
        color=color,
        linewidth=1,
        edgecolor=edge_color,
    )


def save_estimates_to_file(exs, path):
    df = pd.DataFrame(data=map(lambda x: [x.n, x.s], exs))
    df.to_csv(path)


if __name__ == "__main__":
    params = {"alpha": 0.3271620446656849, "beta": 0.8511102671654379}  # ReadIt - US
    # params = { "alpha": 1.1428044756900324, "beta": 3.586332707244498 } # ReadIt - Latin
    # params = {"alpha": 1.07424287184781, "beta": 2.358619301400822} #ReadIt - Italy
    probs = (1, )
    N = 10
    daily_installs = [1000] * 30
    conversion_rate = 0.06
    free_trial_conversion = 0.6
    days = [16, 20, 23, 27]
    true_Ex = BGModel().fitter.static_expected_number_of_purchases_up_to_time(params['alpha'], params['beta'], 52) + 1
    exss = []
    for index, T in enumerate(days):
        print("STARTING: ")
        ex = get_estimates(
            params=params,
            probs=probs,
            daily_installs=daily_installs,
            observed_days=T,
            conversion_rate=conversion_rate,
            free_trial_conversion=free_trial_conversion,
            N=N)
        exss.append(ex)

    set_plot_title(true_Ex, N, daily_installs=daily_installs[0],conversion_rate=conversion_rate, free_trial_conversion=free_trial_conversion)
    for index, exs in enumerate(exss):
        print_estimates_lines(exs, true_Ex, days[index])
    plt.show()

    set_plot_title(true_Ex)
    for index, exs in enumerate(exss):
        print_estimates_lines(exs, true_Ex, days[index])
    plt.show()

    for index, exs in enumerate(exss):
        save_estimates_to_file(exs, '~/data{}.csv'.format(index))

