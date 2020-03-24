# Sense2Stop: Smoking detection via hierarchical latent marked point processes  #

This project is statistical mdoels for detection of smoking episodes using self-report and sensors to measure smoking episodes.  Time-varying covariates such as urge and stress are measured via self-report.  Interventions are provided over the study in order to reduce proximal stress.  The proposed model attempts to account for the uncertainty in event times when jointly modeling risk and time-varying health processes (e.g., urge/stress) as well as in assessing impact of intervention on proximal risk of smoking.

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

0. Dependencies. If there are any dependencies list them. 
1. Get the data
* Where is the data, who is in charge of it, how do they get it. 
* Are there preprocessing steps? If so what folder are they in, for example: [the data preprocessing directory](/data_preprocessing)
2. Run methods. Point people towards the folder with methods. [the methods directory](/methods)
3. Evaluate. Point people towards the folder with evaluation functions [the evaluation directory](/evaluation)

# Notes #

To see more tips on README's see [here](https://github.com/tchapi/markdown-cheatsheet/blob/master/README.md)

SOMETHING TO NOTE: Readmes are incredibly sensitive to spaces, if you are not sure why something isn't working double check the example and make sure you have the spacing right. 

Here is an example code block:

```
mvn dependency:build-classpath -Dmdep.outputFile=classpath.out
```
