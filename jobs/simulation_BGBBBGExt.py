import pytest
import math
import lifetimes.generate_data as gen
import numpy as np
import lifetimes.estimation as est
import lifetimes.models as mod
from lifetimes.data_compression import compress_data, compress_session_session_before_conversion_data
from lifetimes.data_compression import filter_data_by_T
from lifetimes import models
import pandas as pd
from lifetimes.estimation import BGBBFitter, BGBBBGExtFitter


@pytest.mark.sim_BGBBBBExt
def test_BGBBBGExt_fitting_on_simulated_quite_real_looking_data():
    T = 7
    T_lagged = 6
    T0 = 52

    sizes_installs = [500, 1000, 5000, 10000, 25000, 50000] # , 100000]  # [500, 1000]  #
    n_sim = 10
    iterative_fitting = 0
    penalizer_coef = 0.1

    success_fit_ratio = {}
    success_fit = {}
    arpd_estimates = {}
    lifetime_estimates = {}
    conversion_estimates = {}
    appd_estimates = {}
    apppu_estimates = {}
    arppu_estimates = {}

    for size_installs in sizes_installs:

        success_fit[size_installs] = 0
        arpd_estimates[size_installs] = []
        lifetime_estimates[size_installs] = []
        conversion_estimates[size_installs] = []
        appd_estimates[size_installs] = []
        apppu_estimates[size_installs] = []
        arppu_estimates[size_installs] = []

        for n in range(n_sim):

            # size_installs = 5000
            size_purchasers = size_installs / 20

            a, b, g, d, e, z, c0, a2, b2, g2, d2 = 1.13, 0.32, 0.63, 3.2, 0.05, 6.78, 0.04, 0.33, 1.75, 1.88, 7.98

            params_conversion = {'alpha': a, 'beta': b, 'gamma': g, 'delta': d, 'epsilon': e, 'zeta': z, 'c0': c0}
            params_arppu = {'alpha': a2, 'beta': b2, 'gamma': g2, 'delta': d2}

            n_cohorts = T - T_lagged + 1

            data_conversion = gen.bgbbbgext_model(T, params_conversion['alpha'], params_conversion['beta'],
                                                  params_conversion['gamma'], params_conversion['delta'],
                                                  params_conversion['epsilon'],
                                                  params_conversion['zeta'], params_conversion['c0'],
                                                  size=size_installs/n_cohorts,
                                                  time_first_purchase=True)

            data_arppu = gen.bgbb_model(T, params_arppu['alpha'], params_arppu['beta'],
                                        params_arppu['gamma'], params_arppu['delta'],
                                        size=size_purchasers/n_cohorts)

            for Ti in range(T_lagged, T):
                data_conversion_new = gen.bgbbbgext_model(Ti, params_conversion['alpha'], params_conversion['beta'],
                                                      params_conversion['gamma'], params_conversion['delta'],
                                                      params_conversion['epsilon'],
                                                      params_conversion['zeta'], params_conversion['c0'],
                                                      size=size_installs / n_cohorts,
                                                      time_first_purchase=True)
                data_arppu_new = gen.bgbb_model(Ti, params_arppu['alpha'], params_arppu['beta'],
                                            params_arppu['gamma'], params_arppu['delta'],
                                            size=size_purchasers / n_cohorts)
                data_conversion = pd.concat([data_conversion, data_conversion_new])
                data_arppu = pd.concat([data_arppu, data_arppu_new])

            mv_values = gen.sample_monetary_values(size_purchasers)

            compressed_data_conversion = compress_session_session_before_conversion_data(data_conversion)
            compressed_data_arppu = compress_data(data_arppu)

            model_conversion = mod.BGBBBGExtModel(penalizer_coef)
            model_conversion.fit(frequency=compressed_data_conversion['frequency'],
                                 recency=compressed_data_conversion['recency'], T=compressed_data_conversion['T'],
                                 frequency_before_conversion=compressed_data_conversion['frequency_before_conversion'],
                                 N=compressed_data_conversion['N'], initial_params=params_conversion.values(),
                                 iterative_fitting=iterative_fitting)

            model_arppu = mod.BGBBModel(penalizer_coef)
            model_arppu.fit(frequency=compressed_data_arppu['frequency'], recency=compressed_data_arppu['recency'],
                            T=compressed_data_arppu['T'],
                            N=compressed_data_arppu['N'], initial_params=params_arppu.values(),
                            iterative_fitting=iterative_fitting)

            mv, mv_err = np.mean(mv_values), np.std(mv_values) / math.sqrt(len(mv_values))

            print "Conversion parameters"
            print params_conversion
            print model_conversion.params

            print "Arppu parameters"
            print params_arppu
            print model_arppu.params

            print "Monetary values"
            print mv, mv_err

            ts = range(T0)
            lifetime = [model_conversion.expected_number_of_sessions_up_to_time_with_errors(t) for t in ts]
            conversion_diff = [model_conversion.expected_probability_of_converting_at_time_with_error(t) for t in ts]
            conversion = [model_conversion.expected_probability_of_converting_within_time_with_error(t) for t in ts]
            apppu = [model_arppu.expected_number_of_purchases_up_to_time_with_errors(t) for t in ts]
            arppu = [((1 + apppu[i][0]) * mv,
                      (1 + apppu[i][0]) * mv * math.sqrt((apppu[i][1] / apppu[i][0]) ** 2 + (mv_err / mv) ** 2)) for
                     i in range(len(apppu))]
            appd = [get_arpd_retention_with_error(model_conversion, model_arppu, t) for t in ts]
            arpd = [(appd[i][0] * mv,
                     appd[i][0] * mv * math.sqrt((appd[i][1] / appd[i][0]) ** 2 + (mv_err / mv) ** 2)) for
                    i in range(len(appd))]
            print ts
            print lifetime
            print conversion_diff
            print conversion
            print apppu
            print arppu
            print appd
            print arpd

            summary_df = pd.DataFrame({
                'lifetime': [v[0] + 1 for v in lifetime],
                'lifetime_err': [v[1] for v in lifetime],
                'conversion_diff': [v[0] for v in conversion_diff],
                'conversion_diff_err': [v[1] for v in conversion_diff],
                'conversion': [v[0] for v in conversion],
                'conversion_err': [v[1] for v in conversion],
                'apppu': [v[0] + 1 for v in apppu],
                'apppu_err': [v[1] for v in apppu],
                'arppu': [v[0] for v in arppu],
                'arppu_err': [v[1] for v in arppu],
                'appd': [v[0] for v in appd],
                'appd_err': [v[1] for v in appd],
                'arpd': [v[0] for v in arpd],
                'arpd_err': [v[1] for v in arpd],
                'true_lifetime': [get_true_lifetime(a, b, g, d, t) for t in ts],
                'true_conversion': [get_true_conversion(a, b, g, d, e, z, c0, t) for t in ts],
                'true_apppu': [get_true_apppu(a2, b2, g2, d2, t) for t in ts],
                'true_arppu': [get_true_arppu(a2, b2, g2, d2, true_mv, t) for t in ts],
                'true_appd': [get_true_appd(a, b, g, d, e, z, c0, a2, b2, g2, d2, t) for t in ts],
                'true_arpd': [get_true_arpd(a, b, g, d, e, z, c0, a2, b2, g2, d2, true_mv, t) for t in ts],
            })

            with open("/Users/marcomeneghelli/Desktop/arpd_simulations/" + str(size_installs) + "/pars_simdata_" + str(
                    size_installs) + "_" + str(
                T) + "iterative_fitting" + str(iterative_fitting) + "_" + str(n) + ".txt", "w") as text_file:
                text_file.write(str(model_conversion.params))
                text_file.write(str(model_arppu.params))
                text_file.write(str((mv, mv_err)))

            summary_df.to_csv(
                "/Users/marcomeneghelli/Desktop/arpd_simulations/" + str(size_installs) + "/arpd_simdata_" + str(
                    size_installs) + "_" + str(
                    T) + "iterative_fitting" + str(iterative_fitting) + "_" + str(n) + ".csv")

            last_arpd = arpd[-1][0]
            last_lifetime = lifetime[-1][0]
            last_conversion = conversion[-1][0]
            last_appd = appd[-1][0]
            last_apppu = apppu[-1][0]
            last_arppu = arppu[-1][0]

            if not (math.isnan(last_arpd) or last_arpd is None):
                success_fit[size_installs] += 1

            arpd_estimates[size_installs].append(last_arpd)
            lifetime_estimates[size_installs].append(last_lifetime)
            conversion_estimates[size_installs].append(last_conversion)
            appd_estimates[size_installs].append(last_appd)
            apppu_estimates[size_installs].append(last_apppu)
            arppu_estimates[size_installs].append(last_arppu)

        success_fit_ratio[size_installs] = float(success_fit[size_installs]) / n_sim

        summary_size_installs_df = pd.DataFrame({
            'success_fit_ratio': success_fit_ratio[size_installs],
            'arpd_estimates': arpd_estimates[size_installs],
            'lifetime_estimates': lifetime_estimates[size_installs],
            'conversion_estimates': conversion_estimates[size_installs],
            'appd_estimates': appd_estimates[size_installs],
            'apppu_estimates': apppu_estimates[size_installs],
            'arppu_estimates': arppu_estimates[size_installs],
        })

        summary_size_installs_df.to_csv(
            "/Users/marcomeneghelli/Desktop/arpd_simulations/arpd_simdata_last_measurements_" + str(
                size_installs) + "_" + str(
                T) + "iterative_fitting" + str(iterative_fitting) + ".csv")


def get_true_lifetime(a, b, g, d, t):
    return 1 + BGBBFitter.static_expected_number_of_purchases_up_to_time(a, b, g, d, t)


def get_true_conversion(a, b, g, d, e, z, c0, t):
    conversion = 0
    for ti in range(t + 1):
        conversion += BGBBBGExtFitter.static_expected_probability_of_converting_at_time(a, b, g, d, e, z, c0, ti)
    return conversion


def get_true_apppu(a, b, g, d, t):
    return 1 + BGBBFitter.static_expected_number_of_purchases_up_to_time(a, b, g, d, t)


def get_true_arppu(a, b, g, d, mv, t):
    return get_true_apppu(a, b, g, d, t) * mv


def get_true_appd(a, b, g, d, e, z, c0, a2, b2, g2, d2, t):
    v = 0
    for ti in range(t + 1):
        vc = BGBBBGExtFitter.static_expected_probability_of_converting_at_time(a, b, g, d, e, z, c0, ti)
        va = BGBBFitter.static_expected_number_of_purchases_up_to_time(a2, b2, g2, d2, t - ti)
        if vc == 0:
            break
        v += (va + 1) * vc
    return v


def get_true_arpd(a, b, g, d, e, z, c0, a2, b2, g2, d2, mv, t):
    return get_true_appd(a, b, g, d, e, z, c0, a2, b2, g2, d2, t) * mv


true_mv = 8.46


def get_arpd_retention_with_error(model_conversion, model_arppu, t):
    """
    Convolves model_conversion + model_arppu to get arpd retention
    Args:
        model_conversion:
        model_arppu:
        t:

    Returns:
    """
    v = 0
    e = 0
    for ti in range(t + 1):
        vc, ec = model_conversion.expected_probability_of_converting_at_time_with_error(ti)
        va, ea = model_arppu.expected_number_of_purchases_up_to_time_with_errors(t - ti)
        if vc == 0:
            break
        if va == 0:
            err = ec
        else:
            err = (ea / va + ec / vc) * va * vc
        v += (va + 1) * vc
        e += err ** 2
    return v, e ** 0.5

###test_BGBBBGExt_fitting_compressed_or_not()
