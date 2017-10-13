from dispersion_of_fit_simulation_BG import get_estimates_from_bootstrap
import lifetimes.generate_data as gen
import lifetimes.data_compression as comp
from lifetimes.models import BGModel
import uncertainties
import math


if __name__ == "__main__":
    params = {"alpha": 0.3271620446656849, "beta": 0.8511102671654379}  # ReadIt - US
    # params = { "alpha": 1.1428044756900324, "beta": 3.586332707244498 } # ReadIt - Latin
    # params = {"alpha": 1.07424287184781, "beta": 2.358619301400822} #ReadIt - Italy
    probs = (1,)
    N = 100
    daily_installs = [2000] * 30
    conversion_rate = 0.06
    free_trial_conversion = 0.6
    days = [16, 20, 23, 27]
    true_Ex = BGModel().fitter.static_expected_number_of_purchases_up_to_time(params['alpha'], params['beta'], 52) + 1
    exss = []
    fitted_e_x = []
    observed_days = 27
    for n in range(N):
        print(n)
        Ts = reduce(lambda x, y: x + y,
                    [[(observed_days - day) / 7] * int(math.floor(installs * conversion_rate * free_trial_conversion))
                     for day, installs in enumerate(daily_installs)])
        Ts = filter(lambda x: x > 0, Ts)
        model = BGModel()
        gen_data = gen.bgext_model(Ts, params['alpha'], params['beta'], probs=probs)
        data = comp.compress_bgext_data(gen_data)
        model.fit(frequency=data["frequency"], T=data["T"], N=data["N"], bootstrap_size=100)
        ex = model.expected_number_of_purchases_up_to_time(52) + 1
        fitted_e_x.append(ex)

    contained = len(filter(lambda x: x.n -x.s < true_Ex < x.n+x.s, fitted_e_x)) * 100.0 / N
    print("Interval contains true value: %{} of times".format(contained))