# Measurement-error model (Mem) development

This folder contains a series of ipython notebooks that perform various checks to inform the 


1. How well does HTMGs cover smoking events reported in other measurements? 
    +  `contingent_htmgs.ipynb`: 
    +  `eod_htmgs.ipynb`: 
    +  `random_htmgs.ipynb`:
2. How reliable are events reported in eod_ema in tracking smoking events reported in other measurements?
    + `sr_eod.ipynb`: Recall of 
    + `sr_eod.ipynb`: Recall of 
    + `sr_eod.ipynb`: Recall of 
    + Suggests if the user had an event AND reports in the evening, then the plus/minus on this report is within a 1 hour window.
3. How reliable is puffMarker? We compare when both self-report and random EMA responses to puffMarker.  
    + `sr_puffmarker.ipynb`: For every self-report of 'Yes', we ask if 
    + `random_puffmarker`: 
    + Suggest false-positive:
    + Suggests false-negative: