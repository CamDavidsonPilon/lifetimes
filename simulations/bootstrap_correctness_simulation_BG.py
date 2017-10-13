from dispersion_of_fit_simulation_BG import get_estimates_from_bootstrap
import lifetimes.generate_data as gen
import lifetimes.data_compression as comp
from lifetimes.models import BGModel
from lifetimes.estimation import BGFitter
import math
import numpy as np
import random


if __name__ == "__main__":
    # params = {"alpha": 0.3271620446656849, "beta": 0.8511102671654379}  # ReadIt - US
    # params = { "alpha": 1.1428044756900324, "beta": 3.586332707244498 } # ReadIt - Latin
    params = {"alpha": 1.07424287184781, "beta": 2.358619301400822} #ReadIt - Italy
    probs = (1,)
    N = 100
    daily_installs = [2000] * 30
    conversion_rate = 0.06
    free_trial_conversion = 0.6
    days = [16, 20, 23, 27]
    true_Ex = BGModel().fitter.static_expected_number_of_purchases_up_to_time(params['alpha'], params['beta'], 52) + 1
    exss = []
    fitted_e_x = []
    percentiles_e_x = []
    observed_days = 27
    for n in range(N):
        print(n)
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
        p = current_model.sampled_parameters
        percentiles_data = [BGFitter.static_expected_number_of_purchases_up_to_time(pars['alpha'], pars['beta'], 52) + 1 for pars in current_model.sampled_parameters]
        percentiles_e_x.append((np.percentile(percentiles_data, 16), np.percentile(percentiles_data, 84)))


    contained = len(filter(lambda x: x.n -x.s < true_Ex < x.n+x.s, fitted_e_x)) * 100.0 / N
    print("Interval contains true value: %{} of times".format(contained))
    percentile_contained = len(filter(lambda x: x[0] < true_Ex < x[1], percentiles_e_x)) * 100.0 / N
    print("Percentile interval contains true value: %{} of times".format(percentile_contained))