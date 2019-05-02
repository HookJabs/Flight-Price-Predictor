#Josh Kabo 2019
#Flight Pricing Predictor

from __future__ import print_function

import pandas as pd
pd.__version__

airfare_report_dataframe = pd.read_csv("airfare_report.csv", sep=",")

print("\n\ndf.describe:\n\n")
print(airfare_report_dataframe.describe())

print("\n\nSorted by fare\n\n")
print(airfare_report_dataframe.sort_values('fare', ascending=True))

print("\n\nSliced to just the origin and destination\n\n")
print(airfare_report_dataframe[['origin', 'destination']])

#TODO: Figure out the other columns like fare_lg, lf_ms, etc. Trim these columns out if necessary.