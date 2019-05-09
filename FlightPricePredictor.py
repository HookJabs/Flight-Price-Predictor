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

def my_input_fn(features, targets, batch_size = 1, shuffle = True, num_epochs = None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    #convert pandas data into a dict of np arrays
    features = {key:np.array(value) for key,value in dict(features).items()}

    #construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets)) #NOTE: 2 GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    #shuffle the data if specified
    if shuffle:
        ds = ds.shuffle(buffer_size = 10000)

    #return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


airfare_report_dataframe = pd.read_csv("airfare_report.csv", sep=",")

# print("\n\ndf.describe:\n\n")
# print(airfare_report_dataframe.describe())

# print("\n\nSorted by fare\n\n")
# print(airfare_report_dataframe.sort_values('fare', ascending=True))

# print("\n\nSliced to just the cities\n\n")
# print(airfare_report_dataframe[['city1', 'city2']])

airfare_report_dataframe = airfare_report_dataframe.reindex(np.random.permutation(airfare_report_dataframe.index))
print(airfare_report_dataframe.describe())

#define the input feature, number of miles
my_feature = airfare_report_dataframe[["nsmiles"]]

#configure a numeric feature column for miles
feature_columns = [tf.feature_column.numeric_column("nsmiles")]

#define my label, fare
targets = airfare_report_dataframe["fare"]

#use gradient descent as the optimizer for training the model.
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

#configure the linear regression model with the feature columns and optimizer
#set a learning rate of 0.0000001 for gradient descent
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps = 100
    )


#create an input function for predictions
#it's one prediction per example, so it is unnecessary to reshuffle
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs = 1, shuffle = True)

#call predict on linear regressor to make predictions
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

#format predictions as a NumPy array so we can calculate error metrics
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

#TODO: configure one-hot encoding for locations or something similar to a lat lon implementation
#TODO: split up data into training, test, verification, data.