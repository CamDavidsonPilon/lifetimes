# Changelog

### 0.10.1
 - performance improvements to `generate_data.py` for large datasets #195
 - performance improvements to `summary_data_from_transaction_data`, thanks @MichaelSchreier
 - Previously, `GammaGammaFitter` would have an infinite mean when its `q` parameter was less than 1. This was possible for some datasets. In 0.10.1, a new argument is added to `GammaGammaFitter` to constrain that `q` is greater than 1. This can be done with `q_constraint=True` in the call to `GammaGammaFitter.fit`. See issue #146. Thanks @vruvora 
 - Stop support of scipy < 1.0.
 - Stop support of < Python 3.5.

### 0.10.0
 - `BetaGeoBetaBinomFitter.fit` has replaced `n_custs` with the more appropriately named `weights` (to align with other statisical libraries). By default and if unspecified, `weights` is equal to an array of 1s. 
 - The `conditional_` methods on `BetaGeoBetaBinomFitter` have been updated to handle exogenously provided recency, frequency and periods. 
 - Performance improvements in `BetaGeoBetaBinomFitter`. `fit` takes about 50% less time than previously. 
 - `BetaGeoFitter`, `ParetoNBDFitter`, and `ModifiedBetaGeoFitter` both have a new `weights` argument in their `fit`. This can be used to reduce the size of the data (collapsing subjects with the same recency, frequency, T). 
 
### 0.9.1
 - Added a data generation method, `generate_new_data` to `BetaGeoBetaBinomFitter`. @zscore
 - Fixed a bug in `summary_data_from_transaction_data` that was casting values to `int` prematurely. This was solved by including a new param `freq_multiplier` to be used to scale the resulting durations. See #100 for the original issue.  @aprotopopov
 - Performance and bug fixes in `utils.expected_cumulative_transactions`. @aprotopopov
 - Fixed a bug in `utils.calculate_alive_path` that was causing a difference in values compared to `summary_from_transaction_data`. @DaniGate

### 0.9.0
 - fixed many of the numpy warnings as the result of fitting
 - added optional `initial_params` to all models
 - Added `conditional_probability_of_n_purchases_up_to_time` to `ParetoNBDFitter`
 - Fixed a bug in `expected_cumulative_transactions` and `plot_cumulative_transactions`

### 0.8.1
 - adding new `save_model` and `load_model` functions to all fitters. This will save the model locally as a pickle file.
 - `observation_period_end` in `summary_data_from_transaction_data` and `calibration_and_holdout_data` now defaults to the max date in the dataset, instead of current time. 
 - improved stability of estimators. 
 - improve Runtime warnings. 
 - All fitters are now in a local file. This doesn't change the API however. 