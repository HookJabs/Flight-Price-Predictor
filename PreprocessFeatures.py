#Josh Kabo 2019
#Flight Pricing Predictor

from __future__ import print_function
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
pd.__version__

def intializeDataSet(airfare_report_dataframe):

    #Limits to results from 2016 and later, cutting 80,000 rows. The grand total is now 9001 rows.
    airfare_report_dataframe = airfare_report_dataframe[airfare_report_dataframe.Year >= 2016]

    #starts city market id at 0 
    airfare_report_dataframe['citymarketid_1'] -= 30135

    #the first index here is at 52, but I think it would be unreasonable to refer to the same location as a different number for destination and origin
    airfare_report_dataframe['citymarketid_2'] -= 30135

    #randomizes my data's ordering.
    airfare_report_dataframe = airfare_report_dataframe.reindex(
        np.random.permutation(airfare_report_dataframe.index))
    return airfare_report_dataframe

def preprocess_features(airfare_report_dataframe):
  
  """Prepares input features from airfare_report data set.

  Args:
    airfare_report_dataframe: A Pandas DataFrame expected to contain data
      from the airfare_report data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """

  selected_features = airfare_report_dataframe[
    ["Year",
     "quarter",
     "citymarketid_1",
     "citymarketid_2",
     "nsmiles",
     "passengers",
     #the values below may be good or bad for my results.
     #"carrier_lg",
     "large_ms",
     #"carrier_low",
     "lf_ms"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature to add to the existing features
  processed_features["cost_per_mile"] = (
    airfare_report_dataframe["fare"] /
    airfare_report_dataframe["nsmiles"])
  return processed_features

def preprocess_targets(airfare_report_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    airfare_report_dataframe: A Pandas DataFrame expected to contain data
      from the airfare_report data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["fare"] = (
    airfare_report_dataframe["fare"])
  return output_targets