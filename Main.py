#Josh Kabo 2019
#Flight Pricing Predictor
from __future__ import print_function
import math
from trainModel import train_model
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from PreprocessFeatures import preprocess_features, preprocess_targets, intializeDataSet




airfare_report_dataframe = pd.read_csv("airfare_report.csv", sep=",")

#covers things like resizing, randomizing, and formatting features
airfare_report_dataframe = intializeDataSet(airfare_report_dataframe)

#stores the number of rows in my dataframe
dfSize = len(airfare_report_dataframe.index)

#training data will use the first 7/1oths of the dataframe
training_examples = preprocess_features(airfare_report_dataframe.head(math.floor(dfSize * 0.7)))
training_targets = preprocess_targets(airfare_report_dataframe.head(math.floor(dfSize * 0.7)))

#test data will use the remaining 3/10s of the dataframe
test_examples = preprocess_features(airfare_report_dataframe.tail(math.floor(dfSize * 0.3)))
test_targets = preprocess_targets(airfare_report_dataframe.tail(math.floor(dfSize * 0.3)))

train_model(
    learning_rate=0.00002,
    steps=500,
    batch_size=4
)

