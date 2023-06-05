## What is this project?
This is an exercise in the context of the subject “MINING FROM MASSIVE DATASETS”, that take place in the master’s degree program "Data and Web Science" of Aristotle University of Thessaloniki.

## Purpose
The purpose of this project is to try to evaluate different outlier detection techniques on time series using the exahlon framework (https://github.com/exathlonbenchmark/exathlon)

## Data
The data concerns records from traces recorded from repeated execution of Spark applications. They consist of 9 traces with 3 of them containing known anomalies.
The training set consists of the 6 normal traces and the test set with the 3 traces that contains anomalies.
The known anomalies are the following:
* Bursty Input (T1)
* Bursty Input Until Crash (T2)
* Stalled Input (T3) 
* CPU Contention (T4)
* Process Failure (T5)