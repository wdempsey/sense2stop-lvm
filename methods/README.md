# Measurement-Error Models (MEMs): Methods

## Time Variables
In the scripts in this folder, the variable `study_day` represents the number of days elapsed since participant entry into the study at Day 0 (index begins at zero). In contrast, the variable `day_since_quit` represents the number of days before or after Quit Date: `day_since_quit` is equal to 0 if a given day corresponds to Quit Date itself, while a negative or positive value corresponds to number of days before or after Quit Date, respectively. The variable `is_post_quit` is simply a binary variable equal to 0 if `day_since_quit` is negative (before Quit Date), and equal to 1 if `day_since_quit` is zero or positive (on or after Quit Date). `day_within_period` is the number of days elapsed since beginning of study (for observations recorded before Quit Date) or number of days elapsed since quit day (for observations recorded on or after Quit Date); this variable was created so that time index within each period begins at zero.

## Models
| <img height=0 width=1000> File Name <img height=0 width=1000> | <img height=0 width=1000> Brief Description <img height=0 width=1000> |
|:-----------------------------:|:-----------------------------------------------------------------------|
| [`simple-models-sr.py`](https://github.com/wdempsey/sense2stop-lvm/blob/master/methods/simple-models-sr.py) | Using self-report (SR) data only, three log-linear models of mean counts (`mu`) were considered: (1) `log(mu) = beta` (2) `log(mu) = beta_all + beta_preq * I(is_post_quit=0) + beta_postq * I(is_post_quit=1)` (3) `log(mu) = beta_all + (beta_preq_0 + beta_preq_1*day_within_period) * I(is_post_quit=0) + (beta_postq_0 + beta_postq_1*day_within_period) * I(is_post_quit=1)`|
| [`randeff-models-sr.py`](https://github.com/wdempsey/sense2stop-lvm/blob/master/methods/randeff-models-sr.py) | Using self-report (SR) data only, two log-linear models of mean counts for the `i`th participant (`mu_i`) were considered: (1) `log(mu_i) = gamma_i` (2) `log(mu_i) = beta_all + (gamma_preq_i + beta_preq) * I(is_post_quit=0) + (gamma_postq_i + beta_postq) * I(is_post_quit=1)`|

