import pyautoweka

#Create an experiment
experiment = pyautoweka.ClassificationExperiment(tuner_timeout=360)

import numpy as np
import pandas as pd

X_train = pd.read_csv("./TomWilliams/x_train_IoT_Botnet.csv")
X_test = pd.read_csv("./TomWilliams/x_test_IoT_Botnet.csv")
y_train = pd.read_csv("./TomWilliams/y_train_IoT_Botnet.csv")
y_test = pd.read_csv("./TomWilliams/y_test_IoT_Botnet.csv")

import numpy as np
import pandas as pd
X = pd.read_csv("./TomWilliams/train_IoT_Botnet.csv")
y = X["attack"]

#split into train and test set:
X_train = X[0:100]
y_train = y[0:100]

X_test = X[100:]
y_test = y[100:]

#now we can fit a model:
experiment.fit(X_train, y_train)

#and regress on held out test data:
y_predict = experiment.predict(X_test)
