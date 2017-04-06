import pytest
import lifetimes.generate_data as gen
import lifetimes.models as mod
import lifetimes.data_compression as comp
import math


@pytest.mark.model_fitting
def test_consistence_over_T():
    params = {'r': 1, 'alpha': 10, 's': 0.8, 'beta': 5}

    gen_data = gen.generate_pareto_data_for_T_N(60, 100, params)

    # compressed_gen_data = comp.compress_data(gen_data)

    fitting_window = [(0, 30), (15, 45), (30, 60)]

    t = 365  # time horizon

    fitted_pars = []
    fitted_Ex = []

    for fw in fitting_window:
        pareto_model = mod.ParetoNBDModel()

        filtered_data = comp.filter_data_by_T(gen_data, fw[0], fw[1])

        pareto_model.fit(filtered_data['frequency'], filtered_data['recency'],
                         filtered_data['T'], bootstrap_size=10)

        fitted_pars.append((pareto_model.params, pareto_model.params_C))
        Ex = pareto_model.expected_number_of_purchases_up_to_time(t)
        fitted_Ex.append( (Ex.n, Ex.s)  )

    ex_0, ex_err_0 = fitted_Ex[0]
    for ex, ex_err in fitted_Ex:
        assert math.fabs(ex - ex_0) / math.sqrt(ex_err ** 2 + ex_err_0 ** 2) < 5
