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
from InputFunctions import multi_feature_input_fn, single_input_fn
from PreprocessFeatures import preprocess_features, preprocess_targets, intializeDataSet


#skiprows skips the first row, the labels for the data
airfare_report_dataframe = pd.read_csv("airfare_report.csv", sep=",")


#stores the number of rows in my dataframe
dfSize = len(airfare_report_dataframe.index)


userInput = ""
found = airfare_report_dataframe[airfare_report_dataframe['city1'].str.contains("ert")]
try:
    print(found.iloc[[0],[2]].iat[0,0] == None)
except IndexError:
    found = airfare_report_dataframe[airfare_report_dataframe['city2'].str.contains("Bert")]
    try:
        print(found.iloc[[0],[3]].iat[0,0] == None)
    except IndexError:
        print("location not found.")

# while(userInput.capitalize() != "QUIT"):
#     # print("Input an origin location:")
#     # print("EX: \"Bellingham, WA\"")
#     # userInput = input("")
#     # id_1Input = userInput
#     id_1Input = "Bellingham, WA"

#     # print("Input a destination:")
#     # print("EX: \"Las Vegas, NV\"")
#     # userInput = input("")
#     # id_2Input = userInput
#     id_2Input = "Las Vegas, NV"
    

#     # print("Input about how many miles from origin to destination:")
#     # print("EX: \"1000\"")
#     # userInput = input("")
#     # try:
#     #     userInput = int(userInput)
#     # except ValueError:
#     #     print("invalid number")
#     # milesInput = userInput
#     milesInput = 1000

#     # print("Input the year you plan on traveling:")
#     # print("EX: \"2019\"")
#     # userInput = input("")
#     # try:
#     #     userInput = int(userInput)
#     # except ValueError:
#     #     print("invalid number")
#     # yearInput = userInput
#     yearInput = 2019

#     # print("Input the quarter (season) you plan on traveling:")
#     # print("EX: \"1\"")
#     # userInput = input("")
#     # try:
#     #     userInput = int(userInput)
#     # except ValueError:
#     #     print("invalid number")
#     # quarterInput = userInput
#     quarterInput = 1

    