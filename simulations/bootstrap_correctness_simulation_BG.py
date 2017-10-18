from dispersion_of_fit_simulation_BG import print_estimates_hist
import lifetimes.generate_data as gen
import lifetimes.data_compression as comp
from lifetimes.models import BGModel
from lifetimes.estimation import BGFitter
import math
import numpy as np
import matplotlib.pyplot as plt
import uncertainties


parameters = {
    "US": {"alpha": 0.3271620446656849, "beta": 0.8511102671654379},
    "Italy": {"alpha": 1.07424287184781, "beta": 2.358619301400822},
    "Decreasing": {"alpha": 0.5, "beta": 4},
    "Bell": {"alpha": 3, "beta": 3 },
    "Increasing": {"alpha": 2, "beta": 0.9 }
}


def print_and_add_to_result(string, result):
    print(string)
    result.append(string)

if __name__ == "__main__":
    # params = {"alpha": 0.3271620446656849, "beta": 0.8511102671654379}  # ReadIt - US
    # params = { "alpha": 1.1428044756900324, "beta": 3.586332707244498 } # ReadIt - Latin
    # params = {"alpha": 1.07424287184781, "beta": 2.358619301400822} # ReadIt - Italy
    # params = {"alpha": 0.5, "beta": 4} # decreasing
    # params = {"alpha": 3, "beta": 3 } # bell
    # params = {"alpha": 2, "beta": 0.9 } # increasing
    result = []
    for description, params in parameters.iteritems():
        print_and_add_to_result("{}:".format(description), result)
        for observed_days in [16, 20, 23, 27]:
            print_and_add_to_result("{} days:".format(observed_days), result)
            probs = (1,)
            N = 100
            daily_installs = [2000] * 30
            conversion_rate = 0.06
            free_trial_conversion = 0.6
            true_Ex = BGModel().fitter.static_expected_number_of_purchases_up_to_time(params['alpha'], params['beta'], 52) + 1
            exss = []
            fitted_e_x = []
            percentiles_e_x = []
            observed_days = 20
            for n in range(N):
                if ((n+1) % 10) == 0:
                    print(n+1)
                Ts = reduce(lambda x, y: x + y,
                            [[(observed_days - day) / 7] * int(math.floor(installs * conversion_rate * free_trial_conversion))
                             for day, installs in enumerate(daily_installs)])
                Ts = filter(lambda x: x > 0, Ts)
                current_model = BGModel()
                gen_data = gen.bgext_model(Ts, params['alpha'], params['beta'], probs=probs)
                data = comp.compress_bgext_data(gen_data)
                current_model.fit(frequency=data["frequency"], T=data["T"], N=data["N"], bootstrap_size=100)
                ex = current_model.expected_number_of_purchases_up_to_time(52) + 1
                fitted_e_x.append(ex)
                percentiles_data = filter(lambda x: not (math.isnan(x) or math.isinf(x)), [BGFitter.static_expected_number_of_purchases_up_to_time(pars['alpha'], pars['beta'], 52) + 1 for pars in current_model.sampled_parameters])
                if len(percentiles_data) > 0:
                    percentiles = (np.percentile(percentiles_data, 16), np.percentile(percentiles_data, 84))
                    percentiles_e_x.append(percentiles)
                # plt.hist(
                #     percentiles_data,
                #     bins=range(40), normed=0,
                #     alpha=0.3
                # )
                # plt.axvline(x=true_Ex, color='red', alpha =0.7)
                # plt.axvline(x=percentiles[0], color='blue', alpha =0.7)
                # plt.axvline(x=percentiles[1], color='blue', alpha =0.7)
                # plt.axvline(x=ex.n - ex.s, color='green', alpha =0.7)
                # plt.axvline(x=ex.n + ex.s, color='green', alpha =0.7)
                # plt.show()
            fitted_e_x = filter(lambda x: not math.isnan(x.n) and not math.isinf(x.n), fitted_e_x)
            contained = len(filter(lambda x: x.n - x.s <= true_Ex <= x.n + x.s, fitted_e_x)) * 100.0 / len(fitted_e_x)
            out_upper = len(filter(lambda x: x.n - x.s > true_Ex, fitted_e_x)) * 100.0 / len(fitted_e_x)
            out_lower = len(filter(lambda x: x.n + x.s < true_Ex, fitted_e_x)) * 100.0 / len(fitted_e_x)
            print_and_add_to_result("Interval: {}% {}% {}%".format(out_lower, contained, out_upper), result)
            percentile_contained = len(filter(lambda x: x[0] < true_Ex < x[1], percentiles_e_x)) * 100.0 / len(percentiles_e_x)
            percentile_upper = len(filter(lambda x: x[0] > true_Ex, percentiles_e_x)) * 100.0 / len(percentiles_e_x)
            percentile_lower = len(filter(lambda x: x[1] < true_Ex, percentiles_e_x)) * 100.0 / len(percentiles_e_x)
            print_and_add_to_result("Percentile interval: {}% {}% {}%".format(percentile_lower, percentile_contained, percentile_upper), result)
        print_and_add_to_result("", result)
    print("RESULT")
    for str in result:
        print(str)