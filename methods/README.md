# Measurement-Error Models (MEMs): Methods

## Time Variables
In the scripts in this folder, the variable `study_day` represents the number of days elapsed since participant entry into the study at Day 0 (index begins at zero). In contrast, the variable `day_since_quit` represents the number of days before or after Quit Date: `day_since_quit` is equal to 0 if a given day corresponds to Quit Date itself, while a negative or positive value corresponds to number of days before or after Quit Date, respectively. The variable `is_post_quit` is simply a binary variable equal to 0 if `day_since_quit` is negative (before Quit Date), and equal to 1 if `day_since_quit` is zero or positive (on or after Quit Date). `day_within_period` is the number of days elapsed since beginning of study (for observations recorded before Quit Date) or number of days elapsed since quit day (for observations recorded on or after Quit Date); this variable was created so that time index within each period begins at zero.

## Prepared Data
| <img height=0 width=800> File Name <img height=0 width=800> | <img height=0 width=1000> Brief Description <img height=0 width=1000> |
|:-----------------------------:|:-----------------------------------------------------------------------|
| [`setup-data.py`](https://github.com/wdempsey/sense2stop-lvm/blob/master/methods/setup-data.py) | Participant Quit Date data is merged with participant self-report (SR) data so that each row records timestamp of self-reported smoking event with respect to Time Variables. |

## Models
| <img height=0 width=1000> File Name <img height=0 width=1000> | <img height=0 width=1000> Brief Description <img height=0 width=1000> |
|:-----------------------------:|:-----------------------------------------------------------------------|
| [`pp-models-sr.py`](https://github.com/wdempsey/sense2stop-lvm/blob/master/methods/pp-models-sr.py) | Using self-report (SR) data only, time-to-event models were estimated|
| [`simple-models-sr.py`](https://github.com/wdempsey/sense2stop-lvm/blob/master/methods/simple-models-sr.py) | Using self-report (SR) data only, the following log-linear models of mean counts (`mu`) were considered: (1) `log(mu) = beta` (2) `mu = posquit_mu * is_post_quit + prequit_mu*(1 - is_post_quit)` where `log(postquit_mu) = beta_postq` and `log(prequit_mu) = beta_preq`|
| [`randeff-models-sr.py`](https://github.com/wdempsey/sense2stop-lvm/blob/master/methods/randeff-models-sr.py) | Using self-report (SR) data only, the following log-linear models of mean counts for the `i`th participant (`mu_i`) were estimated: (1) `log(mu_i) = beta + gamma_i` (2) `mu_i = posquit_mu_i * is_post_quit + prequit_mu_i*(1 - is_post_quit)` where `log(postquit_mu_i) = gamma_postq_i + beta_postq` and `log(prequit_mu_i) = gamma_preq_i + beta_preq`|

## Other
| <img height=0 width=800> File Name <img height=0 width=800> | <img height=0 width=1000> Brief Description <img height=0 width=1000> |
|:-----------------------------:|:-----------------------------------------------------------------------|
| [`open-picklejar.py`](https://github.com/wdempsey/sense2stop-lvm/blob/master/methods/open-picklejar.py) | Code to open pickled output of scripts in Models |

