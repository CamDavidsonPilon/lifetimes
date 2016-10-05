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


@pytest.mark.BGBBBB
def test_BGBBBG_generation():
    T = 100
    size = 100

    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    gen_data = gen.bgbbbg_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                                params['epsilon'],
                                params['zeta'], size=size)

    assert len(gen_data) == 100
    assert 'T' in gen_data
    assert 'frequency' in gen_data
    assert 'frequency_before_conversion' in gen_data
    assert 'recency' in gen_data
    assert 'p' in gen_data
    assert 'c' in gen_data
    assert 'theta' in gen_data
    assert 'alive' in gen_data


@pytest.mark.BGBBBG
def test_BGBBBG_generation_with_time_first_purchase():
    T = 100
    size = 10000

    params = {'alpha': 0.48, 'beta': 0.96, 'gamma': 0.25, 'delta': 1.3, 'epsilon': 1.0, 'zeta': 10.0}

    gen_data = gen.bgbbbg_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                                params['epsilon'],
                                params['zeta'], size=size, time_first_purchase=True)

    assert 'time_first_purchase' in gen_data

    # build conversion profile
    ts = range(T + 1)
    cs = [0] * len(ts)
    cs_err = [0] * len(ts)
    tfp_column = gen_data['time_first_purchase']
    N = len(tfp_column)
    for t in ts:
        n = float(len(tfp_column[tfp_column == t]))
        p = n / N
        cs[t] = p
        cs_err[t] = np.sqrt(1.0 / N * p * (1.0 - p))

    conversion_data_frame = pd.DataFrame({'t': ts, 'c': cs, 'c_err': cs_err})


@pytest.mark.BGBBBG
def test_likelyhood():
    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    # test simgle number as inputs
    freq = 1
    sess_before_purch = 1
    rec = 1
    T = 2
    ll = est.BGBBBGFitter._negative_log_likelihood(params.values(), freq, rec, T, sess_before_purch, penalizer_coef=0,
                                                   N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test arrays as inputs
    freq = [1, 0]
    sess_before_purch = [1, 1]
    rec = [1, 0]
    T = [2, 2]
    ll = est.BGBBBGFitter._negative_log_likelihood(params.values(), freq, rec, T, sess_before_purch, penalizer_coef=0,
                                                   N=None)

    assert isinstance(ll, float) or isinstance(ll, int)

    # test np.arrays as inputs
    freq = np.array([1, 0])
    sess_before_purch = np.array([1, 1])
    rec = np.array([1, 0])
    T = np.array([2, 2])
    ll1 = est.BGBBBGFitter._negative_log_likelihood(params.values(), freq, rec, T, sess_before_purch, penalizer_coef=0,
                                                    N=None)

    assert ll == ll1
    assert isinstance(ll1, float) or isinstance(ll1, int)


@pytest.mark.BGBBBB
def test_BGBBBG_fitting_compressed_or_not():
    T = 30
    size = 10000

    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    data = gen.bgbbbg_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                            params['epsilon'],
                            params['zeta'], size=size, time_first_purchase=True)

    compressed_data = compress_session_session_before_conversion_data(data)

    # fitter = est.BGBBBGFitter()
    fitter_compressed = est.BGBBBGFitter()

    # fitter.fit(data['frequency'], data['recency'], data['T'], data['frequency_before_conversion'],
    #           initial_params=params.values())
    fitter_compressed.fit(compressed_data['frequency'], compressed_data['recency'], compressed_data['T'],
                          compressed_data['frequency_before_conversion'],
                          N=compressed_data['N'], initial_params=params.values())

    print params
    # print fitter.params_
    print fitter_compressed.params_
    tot = 0
    fitted_conv = []
    for t in range(31):
        res = fitter_compressed.expected_probability_of_converting_at_time(t)
        tot += res
        fitted_conv.append(res)
    print tot

    conversion_instants = data['time_first_purchase']
    real_conv = [0] * 31
    for t in conversion_instants:
        if not math.isnan(t):
            real_conv[int(t)] += 1.0
    real_conv = [x / len(conversion_instants) for x in real_conv]

    for r, f in zip(real_conv, fitted_conv):
        print r, " - ", f

    print sum(real_conv), " - ", sum(fitted_conv)


    # for par_name in params.keys():
    # assert math.fabs(fitter.params_[par_name] - fitter_compressed.params_[par_name]) < 0.00001


@pytest.mark.BGBBBB
def test_BGBBBG_fitting_compressed_or_not():
    T = 10
    size = 100

    params = {'alpha': 1.2, 'beta': 0.7, 'gamma': 0.6, 'delta': 2.7, 'epsilon': 1.0, 'zeta': 10.0}

    data = gen.bgbbbg_model(T, params['alpha'], params['beta'], params['gamma'], params['delta'],
                            params['epsilon'],
                            params['zeta'], size=size, time_first_purchase=True)

    compressed_data = compress_session_session_before_conversion_data(data)

    # fitter = est.BGBBBGFitter()
    model = mod.BGBBBGModel()
    # fitter.fit(data['frequency'], data['recency'], data['T'], data['frequency_before_conversion'],
    #           initial_params=params.values())
    model.fit(frequency=compressed_data['frequency'], recency=compressed_data['recency'], T=compressed_data['T'],
              frequency_before_conversion=compressed_data['frequency_before_conversion'],
              N=compressed_data['N'], initial_params=params.values())

    print params
    # print fitter.params_
    print model.params
    tot = 0
    fitted_conv = []

    for t in range(30):
        print model.expected_probability_of_converting_at_time_with_error(t)


@pytest.mark.BGBBBB
def test_BGBBBGExt_fitting_on_simulated_quite_real_looking_data():
    T = 7

    sizes_installs = [500, 1000]  # [500, 1000, 5000, 10000, 25000, 50000, 100000]
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

    for size_installs in sizes_installs:

        success_fit[size_installs] = 0
        arpd_estimates[size_installs] = []
        lifetime_estimates[size_installs] = []
        conversion_estimates[size_installs] = []
        appd_estimates[size_installs] = []
        apppu_estimates[size_installs] = []

        for n in range(n_sim):

            # size_installs = 5000
            size_purchasers = size_installs / 20

            params_conversion = {'alpha': 1.13, 'beta': 0.32, 'gamma': 0.63, 'delta': 3.2, 'epsilon': 0.05,
                                 'zeta': 6.78,
                                 'c0': 0.04}
            params_arppu = {'alpha': 0.33, 'beta': 1.75, 'gamma': 1.88, 'delta': 7.98}

            data_conversion = gen.bgbbbgext_model(T, params_conversion['alpha'], params_conversion['beta'],
                                                  params_conversion['gamma'], params_conversion['delta'],
                                                  params_conversion['epsilon'],
                                                  params_conversion['zeta'], params_conversion['c0'],
                                                  size=size_installs,
                                                  time_first_purchase=True)

            data_arppu = gen.bgbb_model(T, params_arppu['alpha'], params_arppu['beta'],
                                        params_arppu['gamma'], params_arppu['delta'],
                                        size=size_purchasers)

            mv_values = gen.sample_monetary_values(size_purchasers)

            compressed_data_conversion = compress_session_session_before_conversion_data(data_conversion)
            compressed_data_arppu = compress_data(data_arppu)

            model_conversion = mod.BGBBBGExtModel()
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

            ts = range(26)
            lifetime = [model_conversion.expected_number_of_sessions_up_to_time_with_errors(t) for t in ts]
            conversion_diff = [model_conversion.expected_probability_of_converting_at_time_with_error(t) for t in ts]
            conversion = [model_conversion.expected_probability_of_converting_within_time_with_error(t) for t in ts]
            apppu = [model_arppu.expected_number_of_purchases_up_to_time_with_errors(t) for t in ts]
            appd = [get_arpd_retention_with_error(model_conversion, model_arppu, t) for t in ts]
            arpd = [(appd[i][0] * mv,
                     appd[i][0] * mv * math.sqrt((appd[i][1] / appd[i][0]) ** 2 + (mv_err / mv) ** 2)) for
                    i in range(len(appd))]
            print ts
            print lifetime
            print conversion_diff
            print conversion
            print apppu
            print appd
            print arpd

            summary_df = pd.DataFrame({
                'lifetime': [v[0] + 1 for v in lifetime],
                'lifetime_err': [v[1] + 1 for v in lifetime],
                'conversion_diff': [v[0] for v in conversion_diff],
                'conversion_diff_err': [v[1] for v in conversion_diff],
                'conversion': [v[0] for v in conversion],
                'conversion_err': [v[1] for v in conversion],
                'apppu': [v[0] + 1 for v in apppu],
                'apppu_err': [v[1] + 1 for v in apppu],
                'appd': [v[0] for v in appd],
                'appd_err': [v[1] for v in appd],
                'arpd': [v[0] for v in arpd],
                'arpd_err': [v[1] for v in arpd],
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

            if not (math.isnan(last_arpd) or last_arpd is None):
                success_fit[size_installs] += 1

            arpd_estimates[size_installs].append(last_arpd)
            lifetime_estimates[size_installs].append(last_lifetime)
            conversion_estimates[size_installs].append(last_conversion)
            appd_estimates[size_installs].append(last_appd)
            apppu_estimates[size_installs].append(last_apppu)

        success_fit_ratio[size_installs] = float(success_fit[size_installs]) / n_sim

        summary_size_installs_df = pd.DataFrame({
            'success_fit_ratio': success_fit_ratio[size_installs],
            'arpd_estimates': arpd_estimates[size_installs],
            'lifetime_estimates': lifetime_estimates[size_installs],
            'conversion_estimates': conversion_estimates[size_installs],
            'appd_estimates': appd_estimates[size_installs],
            'apppu_estimates': apppu_estimates[size_installs],

        })

        summary_size_installs_df.to_csv(
            "/Users/marcomeneghelli/Desktop/arpd_simulations/arpd_simdata_last_measurements_" + str(
                size_installs) + "_" + str(
                T) + "iterative_fitting" + str(iterative_fitting) + ".csv")


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
