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

#print(preprocess_features(airfare_report_dataframe.iloc[0]))


# train_single_feature_linear_regressor(
#     learning_rate=0.00002,
#     steps=500,
#     batch_size=4
# )


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

def makeNewPrediction():
        
    single_example = preprocess_features(airfare_report_dataframe.head(10))
    single_target  = preprocess_targets(airfare_report_dataframe.head(10))
    # training_targets = preprocess_targets(airfare_report_dataframe.iloc[0]))

    predict_more_input_fn = lambda: multi_feature_input_fn(
        single_example, single_target["fare"], 
        num_epochs=1, 
        shuffle=False)


    prediction = linear_regressor.predict(input_fn=predict_more_input_fn)
    predictions = np.array([item['predictions'][0] for item in prediction])
    for p in predictions:
        print(p)
    print("this is it:")
    print(predictions)



makeNewPrediction()

# predictionSet = preprocess_features(airfare_report_dataframe.head(math.floor(dfSize * 0.7)))
# anotherSet = preprocess_features(airfare_report_dataframe.head(math.floor(dfSize * 0.7)))



# single_prediction_input_fn = lambda: single_input_fn(
#     predictionSet, 
#     anotherSet["fare"],
#     batch_size=1)
# print("1")
# #compute new predictions.
# single_prediction = linear_regressor.predict(input_fn=single_prediction_input_fn)
# print("1.5")
# single_prediction = np.array([item['predictions'][0] for item in single_prediction])
# print("2")
# print(single_prediction)

# predict_single_input_fn = lambda: multi_feature_input_fn(
#       test_examples, test_targets["fare"], 
#       num_epochs=1, 
#       shuffle=False)

# single_example = preprocess_features(airfare_report_dataframe.head
# input_fn = tf.estimator.inputs.numpy_input_fn(preprocess_features(airfare_report_dataframe.iloc[0])))
# single_feature = preprocess_features(airfare_report_dataframe.iloc[0])
# single_input_fn = lambda: multi_feature_input_fn(single_feature, single_feature["fare"], batch_size=1)

# single_predictions = linear_regressor.predict(input_fn=single_input_fn)

# print(single_predictions)
# for prediction in single_predictions:
#     print(prediction)
# for single_prediction in SN_classifier.predict(input_fn):
#     predicted_class = single_prediction['class']
#     probability = single_prediction['probability']
#     print(predicted_class + " " + probability)
# predictions = list(linear_regressor.predict(preprocess_features(airfare_report_dataframe.iloc[0])))
# predicted_classes = [p["classes"] for p in predictions]

# print(
#     "New Samples, Class Predictions:    {}\n"
#     .format(predicted_classes))
#print(linear_regressor.predict(preprocess_features(airfare_report_dataframe.iloc[0])))
#TODO: Focus on polish. Presentation is coming up. Use it to make predictions.
#TODO: Encode the carrier data from strings to numeric values
#TODO: Consider using a small neural network
#TODO: Consider regularization