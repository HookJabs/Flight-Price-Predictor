#Josh Kabo 2019
#Flight Pricing Predictor
from __future__ import print_function
import math
import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.data import Dataset
from InputFunctions import multi_feature_input_fn
from PreprocessFeatures import preprocess_features, preprocess_targets, intializeDataSet

airfare_report_dataframe = pd.read_csv("airfare_report.csv", sep=",")

def train_multi_feature_linear_regressor(
  learning_rate,
  steps,
  batch_size,
  training_examples,
  training_targets,
  test_examples,
  test_targets):
  """Trains a linear regression model of multiple features.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and test loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      airfare dataframe to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      airfare dataframe to use as target for training.
    test_examples: A `DataFrame` containing one or more columns from
      airfare dataframe to use as input features for test.
    test_targets: A `DataFrame` containing exactly one column from
      airfare dataframe to use as target for test.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

#TODO: Hey, I set this as 1 because I don't want to wait right now. Should be 10

  periods = 1
  steps_per_period = steps / periods
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
  
  #Create input functions.
  training_input_fn = lambda: multi_feature_input_fn(
    training_examples, 
    training_targets["fare"], 
    batch_size=batch_size)
  predict_training_input_fn = lambda: multi_feature_input_fn(
      training_examples, 
      training_targets["fare"], 
      num_epochs=1, 
      shuffle=False)
  predict_more_input_fn = lambda: multi_feature_input_fn(
      training_examples, training_targets["fare"], 
      num_epochs=1, 
      shuffle=False)
  
  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  test_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    #compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])

    print(training_predictions)
    
    test_predictions = linear_regressor.predict(input_fn=predict_more_input_fn)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])
    
    # Compute training and test loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    test_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    test_rmse.append(test_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(test_rmse, label="test")
  plt.legend()


  #plt.show()

  more_predictions = linear_regressor.predict(input_fn=predict_more_input_fn)
  more_predictions = np.array([item['predictions'][0] for item in test_predictions])

  print(more_predictions)

  return linear_regressor


def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
    input_features: The names of the numerical input features to use.
    Returns:
    A set of feature columns
    """ 
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])

# Because we're now working with multiple input features, let's 
# modularize our code for configuring feature columns into a separate function. 
# (For now, this code is fairly simple, as all our features are numeric, 
# but we'll build on this code as we use other types of features in future exercises.)