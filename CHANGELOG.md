# Changelog

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