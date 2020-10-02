import pyautoweka

#Create an experiment
experiment = pyautoweka.ClassificationExperiment()

import numpy as np
import pandas as pd

X_train = pd.read_csv("x_train_MNIST.csv")
X_test = pd.read_csv("x_test_MNIST.csv")
y_train = pd.read_csv("y_train_MNIST.csv")
y_test = pd.read_csv("y_test_MNIST.csv")

#now we can fit a model:
experiment.fit(X_train, y_train.label)

#and regress on held out test data:
y_predict = experiment.predict(X_test)

y_pred = pd.DataFrame(data = y_predict)
y_pred.to_csv("y_pred_MNIST_Auto_Weka.csv")
