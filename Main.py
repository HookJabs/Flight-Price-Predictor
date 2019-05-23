#Josh Kabo 2019
#Flight Pricing Predictor
from __future__ import print_function
import math
from SingleFeatureLinearRegressor import train_single_feature_linear_regressor
from MultiFeatureLinearRegressor import train_multi_feature_linear_regressor
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from PreprocessFeatures import preprocess_features, preprocess_targets, intializeDataSet

#ingores all warnings unless they are fatal
tf.logging.set_verbosity(tf.logging.DEBUG)



#skiprows skips the first row, the labels for the data
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


# train_single_feature_linear_regressor(
#     learning_rate=0.00002,
#     steps=500,
#     batch_size=4
# )


train_multi_feature_linear_regressor(
    learning_rate=0.00003,
    steps=500.0,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    test_examples=test_examples,
    test_targets=test_targets)

#TODO: my objective for today is to train a linear regressor using all my features.
#TODO: Encode the carrier data from strings to numeric values