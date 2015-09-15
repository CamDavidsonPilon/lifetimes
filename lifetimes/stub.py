def _logregress_a(self, r, t):
    p = self._unload_params('p')
    r['log_dev_m'] = np.log(r['m']) - np.log(r['em'])  # log deviation of m on em
    r['log_dev_x'] = np.log(r['x']) - np.log(r['ex'])  # log deviation of x on ex
    r['x0'] = r['log_dev_x']
    r['x1'] = r['T'] * r['log_dev_x']
    r['x2'] = p * r['log_dev_x']

    a0, a1, a2 = ols(r['log_dev_m'], r[['x0', 'x1', 'x2']], intercept=False).beta
    return a1 * p + a2 * r['T'] + a3 * r['t'] + a0


def _r(self, frequency, recency, T, monetary_value, frequency_fitted_model, k):
    df = DataFrame([frequency, recency, T, monetary_value], columns=['x', 'recency', 'T', 'm'])
    df['ex'] = df.apply(
        lambda r: frequency_fitted_model.predict(
            k,
            r['x'],
            r['recency'],
            r['T']
        ),
        axis=1
    )
    df['em'] = self.conditional_expected_average_profit()

    #df['coeffs'] = df.apply(lambda r: ols())
