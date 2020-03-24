# Data pre-processing #

This code focuses on pre-processing of the Sense2Stop data. The
original files were stored as individual zip files in [Box](http://box.com/).

## Code Description ##

There are 3 sources of data:

1. Cloud data: data that was sent to the mCerebrum cloud storage in real-time. 
    + This data can be missing if syncing did not occur properly
2. Phone (backup) data: data that was stored on the phone and uploaded when the study participant returned study equipment at the end of the study
    + This data can be missing if the participant did not return the equipment
3. Alternative storage: the mCerebrum system did not properly store data for participants 223-227.  Therefore, this data was _recovered_, resulting in slightly different storage structure

We performed a [data source comparison](/data_source_comparison) and found that the phone data, if available, was always a superset of the cloud data.  Therefore, we prioritize phone data and then use cloud data when this is not available.  

Running the following sequence of scripts:

```
python unzip_process.py # De
python unzip_backup_process.py 
python unzip_alternative_process.py 
python unzip_combine.py 
```

This will produce a set of `final` csv files located at `/Box/MD2K Northwestern/Processed Data/smoking-lvm-cleaned-data/final`.  These files are:
* `eod-ema-final.csv`: End-of-day EMAs (ecological momentary assessments)
* `eventcontingent-ema-final.csv`: Event-contingent EMAs.  
    + A subset of self-reported smoking times at which EMAs were triggered
* `hq-episodes-final.csv`: Partition of each user-day into high-quality data episodes.  These are windows over which HTMGs and puffMarker can be triggered.
* `puff-episode-final.csv`: PuffMarker detected smoking episodes
* `puff-probability-final.csv`: Hand-to-mouth-gestures (HTMGs) 
* `random-ema-final.csv`: Random EMAs
* `self-report-smoking-final.csv`: Self-reported smoking times



