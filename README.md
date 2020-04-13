# Sense2Stop: Smoking detection via hierarchical latent marked point processes  #

This project is statistical models for detection of smoking episodes using self-report and sensors to measure smoking episodes.  Time-varying covariates such as urge and stress are measured via self-report.  Interventions are provided over the study in order to reduce proximal stress.  The proposed model attempts to account for the uncertainty in event times when jointly modeling risk and time-varying health processes (e.g., urge/stress) as well as in assessing impact of intervention on proximal risk of smoking.

## Project Description ##
This project includes the code needed to reproduce results.  This includes (A) exploratory data analysis, (B) algorithmic development, and (C) application of models to the cleaned datasets. If using this code please cite the paper using the following bibtex: 

```tex
@article{dempsey:2020,
author = {Dempsey, Walter},
title = {Hierarchical point process and multi-scale measurements: data integration for latent recurrent event analysis under uncertainty},
booktitle = {arXiv},
year = {2020}}
```
The goal of this project is to do. 

## Code Description ##

If there are steps to run the code list them as follows: 

0. Dependencies: all code is developed in Python using [Anaconda](https://anaconda.org/about).
* The Anaconda environment can be installed using [bayesian.yml](./bayesian.yml). See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for instructions on creating the environment.  Simply open Anaconda shell, open to github repo and run:
```
conda env create -f bayesian.yml
```
* _Symbolic links_: Please setup symbolic links for linking to the Box directory since the data location can change across OS and systems.  
  + Requires 4 symbolic links
    - 'cleaned_data' maps to location `...\Box\MD2K Northwestern\Processed Data\smoking-lvm-cleaned-data`
    - 'data_streams' maps to location `...\Box\MD2K Northwestern\Processed Data\Data streams`
    - 'data_streams_backup' maps to location `...\Box\MD2K Northwestern\Processed Data\Data streams - phone backup files`
    - 'final_data` maps to location `...\Box\MD2K Northwestern\Processed Data\smoking-lvm-cleaned-data\final`
  + For example: On Windows the following run from home directory will generate the correct symbolic link for `final_data`
  ```
  mklink /d final_data ...\Box\MD2K Northwestern\Processed Data\smoking-lvm-cleaned-data\final
  ```
1. Data access, preprocessing, and exploratory data analysis
* Data is stored on [Box](https://account.box.com/login) and is owned by PI [Bonnie Spring](https://www.feinberg.northwestern.edu/faculty-profiles/az/profile.html?xid=16136).  Access is limited to the study team; however, 
* [Data preprocessing](/data_preprocessing) converts the raw data into a set of data files
* [Exploratory data analysis](/exploratory_data_analysis) is presented as a set of ipython notebooks. Descriptive statistics are used to inform the prior on the measurement-error models using in the analysis phase
2. The [methods directory](/methods) contains all algorithms for MCMC estimation.  Algorithms are developed within the [pymc3](https://docs.pymc.io/).  The algorithm, at a high-level, performs the following
* Sample event times given observations and parameters (using reversible-MCMC adjustment)
* Sample parameters given latent event times (using pyMC3 software) 
3. All evaluation functions can be found in the [the evaluation directory](/evaluation).  In particular, we perform posterior predictive checks to confirm model fit to the data.
4. Final project report can be found in [the write-up directory](/write-up)
