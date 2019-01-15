# Data pre-processing #

This code focuses on pre-processing of the Sense2Stop data. The
original files were stored as individual zip files

We use self-report
and sensors to measure smoking episodes.  Time-varying covariates such
as urge are measured via self-report.  Location (GPS) is measured via
sensors.  Stress is measured both via self-report and sensors.

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
