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

linear_regressor = train_multi_feature_linear_regressor(
    #best: 0.00015, 500, 4, Score: 45.14
    learning_rate=0.00015,
    #overshoots at 0.00025
    steps=500,
    batch_size=4,
    training_examples=training_examples,
    training_targets=training_targets,
    test_examples=test_examples,
    test_targets=test_targets)

avgLargeMS = airfare_report_dataframe.loc[:,"large_ms"].mean()
avgSmallMS = airfare_report_dataframe.loc[:,"lf_ms"].mean()
avgPassengers = airfare_report_dataframe.loc[:,"passengers"].mean()
avgFare = airfare_report_dataframe.loc[:,"fare"].mean()


userInput = ""

def startPrompts():
    while(True):
        print()
        print("Input an origin location:")
        print("EX: \"Bellingham, WA\"")
        userInput = input("")
        print()
        id_1Input = userInput
        #try to get the id value searching origins
        found = airfare_report_dataframe[airfare_report_dataframe['city1'].str.contains(id_1Input)]
        try:
            id_1Input = found.iloc[[0],[2]].iat[0,0]
            print(id_1Input)
        except IndexError:
            #try to get the id value searching destinations
            found = airfare_report_dataframe[airfare_report_dataframe['city2'].str.contains(id_1Input)]
            try:
                id_1Input = found.iloc[[0],[3]].iat[0,0]
                print(id_1Input)
            except IndexError:
                print("location not found.")
                print()
                print("Error, something went wrong, restarting")
                print()
                startPrompts()

        #if I can find that row and column, just look at the value 2 over on left, that's my id num

        print("Input a destination:")
        print("EX: \"Las Vegas, NV\"")
        userInput = input("")
        print()
        id_2Input = userInput
        #try to get the id value searching origins
        found = airfare_report_dataframe[airfare_report_dataframe['city1'].str.contains(id_2Input)]
        try:
            id_2Input = found.iloc[[0],[2]].iat[0,0]
            print(id_2Input)
        except IndexError:
            #try to get the id value searching destinations
            found = airfare_report_dataframe[airfare_report_dataframe['city2'].str.contains(id_2Input)]
            try:
                id_2Input = found.iloc[[0],[3]].iat[0,0]
                print(id_2Input)
            except IndexError:
                print("location not found.")
                print()
                print("Error, something went wrong, restarting")
                print()
                startPrompts()
        

        print("Input about how many miles from origin to destination:")
        print("EX: \"1000\"")
        print()
        userInput = input("")
        try:
            userInput = int(userInput)
        except ValueError:
            print("invalid number")
            print()
            print("Error, something went wrong, restarting")
            print()
            startPrompts()
        milesInput = userInput

        print("Input the year you plan on traveling:")
        print("EX: \"2019\"")
        print()
        userInput = input("")
        try:
            userInput = int(userInput)
        except ValueError:
            print("invalid number")
            print()
            print("Error, something went wrong, restarting")
            print()
            startPrompts()
        yearInput = userInput

        print("Input the quarter (season) you plan on traveling:")
        print("EX: \"1\"")
        print()
        userInput = input("")
        try:
            userInput = int(userInput)
        except ValueError:
            print("invalid number")
            print()
            print("Error, something went wrong, restarting")
            print()
            startPrompts()
        quarterInput = userInput

        #a temporary dictionary to be made into a small dataset to use for a new prediction
        tempDict = {'Year' : [yearInput], 'quarter' : [quarterInput], 'citymarketid_1' : [id_1Input],
        'citymarketid_2' : [id_2Input], 'nsmiles' : [milesInput], 'passengers' : [avgPassengers],
        'fare' : [avgFare], 'large_ms' : [avgLargeMS], 'lf_ms' : [avgSmallMS]}

        single_df = pd.DataFrame(data=tempDict)

        new_example = preprocess_features(single_df)
        new_target = preprocess_targets(single_df)

        predict_new_input_fn = lambda: multi_feature_input_fn(
            new_example, new_target["fare"], 
            num_epochs=1, 
            shuffle=False)


        prediction = linear_regressor.predict(input_fn=predict_new_input_fn)
        predictions = np.array([item['predictions'][0] for item in prediction])
        print()
        try:
            print("Your price is: ")
            print(predictions)
            print("+/- about $45")
        except:
            print()
            print("Error, something went wrong, restarting")
            print()
            startPrompts()
        print()
        #make the defualt passenger number the avg of the whole dataset.
        #same for:
        # "large_ms",
        # "lf_ms",

startPrompts()