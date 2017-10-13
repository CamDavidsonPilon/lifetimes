import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lifetimes.generate_data as gen
import lifetimes.data_compression as comp
from lifetimes.models import BGModel
import uncertainties


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


def get_estimates_from_bootstrap(params, daily_installs, observed_days, conversion_rate, free_trial_conversion, N):
    model = BGModel(penalizer_coef=0.01)
    Ts = reduce(lambda x, y: x + y,
                [[(observed_days - day) / 7] * int(math.floor(installs * conversion_rate * free_trial_conversion))
                 for day, installs in enumerate(daily_installs)])
    Ts = filter(lambda x: x > 0, Ts)
    gen_data = gen.bgext_model(Ts, params['alpha'], params['beta'], probs=probs)
    data = comp.compress_bgext_data(gen_data)
    model.fit(frequency=data["frequency"], T=data["T"], N=data["N"], bootstrap_size=N)
    exs = []
    for i in range(N):
        a = model.sampled_parameters[i]['alpha']
        b = model.sampled_parameters[i]['beta']
        cov = model.params_C
        [a, b] = uncertainties.correlated_values([a, b], cov)
        Ex = model.wrapped_static_expected_number_of_purchases_up_to_time(a, b, 52) + 1
        print (i, Ex)
        exs.append(Ex)
    return (exs, model.expected_number_of_purchases_up_to_time(52) + 1)


def set_plot_title(true_Ex, N, daily_installs, conversion_rate, free_trial_conversion):
    plt.title(
        'Histogram of '
        + str(N)
        + ' estimates (true: {})'.format(round(true_Ex, 2))
        + ' - '
        + str(np.mean(daily_installs) * conversion_rate * free_trial_conversion)
        + ' conv/day')
    plt.xlabel('estimates')
    plt.axvline(x=true_Ex, color="black", label="true")
    plt.legend()
    plt.grid(True)


def print_estimates_lines(exs, true_Ex, number_of_days, color, edge_color):
    percentage_out_lower = len(filter(lambda x: true_Ex > x + x.s, exs)) * 100.0 / len(exs)
    percentage_out_upper = len(filter(lambda x: true_Ex < x - x.s, exs)) * 100.0 / len(exs)
    percentage_in = len(filter(lambda x: x - x.s < true_Ex < x + x.s, exs)) * 100.0 / len(exs)
    estimates = map(lambda x: x.n, exs)
    hist = np.histogram(a=map(lambda x: x.n, exs), bins=range(0, 40, 2))
    plt.plot(hist[1][:-1], hist[0],
             label='Days:{}, Val: {}+/{}, {}%-{}%-{}%'.format(
        number_of_days,
        round(np.mean(estimates), 1),
        round(np.std(estimates), 1),
        round(percentage_out_lower, 1),
        round(percentage_in, 1),
        round(percentage_out_upper, 1)
    ),
             color=color)
    plt.axvline(x=true_Ex.n, color=edge_color, label="fitted{}: {}+/-{}".format(number_of_days, round(true_Ex.n,2), round(true_Ex.s,2)))


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
    plt.axvline(x=true_Ex.n, color=edge_color, label="fitted{}: {}+/-{}".format(number_of_days, round(true_Ex.n,2), round(true_Ex.s,2)))


def save_estimates_to_file(exs, path):
    df = pd.DataFrame(data=map(lambda x: [x.n, x.s], exs))
    df.to_csv(path)


if __name__ == "__main__":
    params = {"alpha": 0.3271620446656849, "beta": 0.8511102671654379}  # ReadIt - US
    # params = { "alpha": 1.1428044756900324, "beta": 3.586332707244498 } # ReadIt - Latin
    # params = {"alpha": 1.07424287184781, "beta": 2.358619301400822} #ReadIt - Italy
    probs = (1, )
    N = 100
    daily_installs = [2000] * 30
    conversion_rate = 0.06
    free_trial_conversion = 0.6
    days = [16, 20, 23, 27]
    true_Ex = BGModel().fitter.static_expected_number_of_purchases_up_to_time(params['alpha'], params['beta'], 52) + 1
    exss = []
    fitted_e_x = []
    for index, T in enumerate(days):
        print("STARTING: ")
        ex = get_estimates_from_bootstrap(
            params=params,
            #probs=probs,
            daily_installs=daily_installs,
            observed_days=T,
            conversion_rate=conversion_rate,
            free_trial_conversion=free_trial_conversion,
            N=N)
        exss.append(ex[0])
        fitted_e_x.append(ex[1])

    for index, exs in enumerate(exss):
        print_estimates_hist(exs,
                             number_of_days=days[index],
                             true_Ex=fitted_e_x[index],
                             color=[(1, 1, 0, 0.5), (0, 1, 0, 0.5), (1, 0, 0, 0.5), (0, 0, 1, 0.5)][index],
                             edge_color=[(1, 1, 0, 1), (0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1)][index],
                             )
    set_plot_title(true_Ex, N, daily_installs=daily_installs[0], conversion_rate=conversion_rate,
                   free_trial_conversion=free_trial_conversion)
    plt.show()

    for index, exs in enumerate(exss):
        print_estimates_lines(exs, number_of_days=days[index],
                              true_Ex=fitted_e_x[index],
                              color=[(1, 1, 0, 0.5), (0, 1, 0, 0.5), (1, 0, 0, 0.5), (0, 0, 1, 0.5)][index],
                              edge_color=[(1, 1, 0, 1), (0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1)][index],
                              )
    set_plot_title(true_Ex, N, daily_installs=daily_installs[0], conversion_rate=conversion_rate,
                   free_trial_conversion=free_trial_conversion)
    plt.show()

    for index, exs in enumerate(exss):
        save_estimates_to_file(exs, '~/data{}.csv'.format(index))

