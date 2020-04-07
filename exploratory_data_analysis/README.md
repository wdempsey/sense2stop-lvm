# Measurement-error models (Mems): Exploratory Data Analysis (EDA)

This folder contains a series of ipython notebooks that compute descriptive statistics to inform measurement-error models.  Since there is not a large prior literature on many of these measurements, EDA is used to construct data-informed priors.  We answer the following questoins

1. How well does HTMGs cover smoking events reported in other measurements? 
    +  `sr_htmgs.ipynb`: For every `Yes` in self-report, computes the number of HTMGs in a window of length delta=5,15,30 
    +  `random_htmgs.ipynb`:
    +  `puffmarker_htmgs.ipynb`:
    +  EDA suggests the average number of HTMGs in the 15-minutes around a smoking event is 2.916 with a standard deviation of 2.787.  
    +  EDA suggests the average number of HTMGs in the 15-minute window prior to the random EMA with `No` response is 1.787 with a standard deviation of 1.238.
    +  For a hierarchical point process model, this suggests smoking increases the joint rate by 1 in a five minute window around the event.  This is inexact, of course, but suggests HTMGs contain some relevant information for pinpointing smoking event times
2. How reliable are events reported in eod_ema in tracking smoking events reported in other measurements?
    + `sr_eod.ipynb`: Recall of self-report.
        - Fraction agreement is 0.536 for current hour only; 0.833 for plus/minus hour
        - So both suggest a Normal prior with variance 42 minutes 
    + `pM_eod.ipynb`: Recall of puffMarker episodes
        - Fraction agreement is 0.222 for current hour only; 0.409 for plus/minus hour
        - So both suggest a Normal prior with variance 110 minutes 
    + `random_eod.ipynb`: Recall of random EMA
        - Fraction agreement is 0.413 for current hour only; 0.712 for plus/minus hour
        - So both suggest a Normal prior with variance 55 minutes 
    + Suggests if the user had an event AND reports in the evening, then the plus/minus on this report is within a 1 hour window.  
    + We model this as given an event at time T, the user _recalls_ an event at time Z which is normally distributed with mean T and variance that is inverse-wishart distributed with mean 45.
    + Future work can adjust model based on dependence on type of other measurement since self-report had higher recall than random EMA which had higher recall than puffMarker, but this was not considered here.
3. How reliable is puffMarker? We compare when both self-report and random EMA responses to puffMarker.  
    + `pM_confirmation.ipynb`: For every self-report of 'Yes', we ask if 
    + (False-positive) 15-minute window the fraction where self-report or random EMA say 'Yes' and pM does not trigger is 0.88
    + (False-negative) 15-minute window the fraction where random EMA say 'No' and pM triggers is 0.017
    + (False-negative) Fraction of days where end-of-day EMA says all 'No' and pM triggers at least once is 0.302
4. How reliable is random EMA?
5. How reliable is self-report?

# Priors for recurrent event analysis: Exploratory Data Analysis (EDA)

6. How does the number of smoking events vary across study-day?
    + `survival.ipynb`: Contains all recurrent event analysis.  
