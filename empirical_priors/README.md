# Empirical Priors

This directory contains scripts used to build the empirical priors for the various parameters in the model.

eod_bias_variance.ipynb computes End-of-day EMA's potential bias + variance as an estimator for the smoking times reported in event-contingent EMAs.

smoking_time_prior.ipynb computes the prior on the time-between smoking events and the rate of smoking as a function of day in study. This portion is still incomplete.

alignment.py, day_alignment.py, day_alignment.ipynb compute summary statistics of the temporal alignments of the participants. Their outputs are stored in alignment_user.csv and alignment_summary.csv. 

empirical_prior_report.pdf is a short report that explains the approaches and findings we come across in the process of building the empirical priors relevant to our study.