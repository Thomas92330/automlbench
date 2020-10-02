print("Begin imports")
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML
from time import process_time
print("Begin h2o")
h2o.init()
df = h2o.upload_file('train_MNIST.csv')
df["label"] = df["label"].asfactor()
response = "label"
predictors=[]
for col in df.columns:
    if col != 'label':
        predictors.append(col)

train = h2o.upload_file('train_MNIST.csv')

valid = h2o.upload_file('y_train_MNIST.csv')

mnist_gbm = H2OGradientBoostingEstimator()

t1_start = process_time()
mnist_gbm.train(x = predictors,
               y = response,
               training_frame = train,
               validation_frame = valid)
t1_stop = process_time()

print( mnist_gbm)

print("Elapsed time in seconds : ",t1_stop-t1_start)  

X_test = h2o.upload_file('x_test_MNIST.csv')
predictions = mnist_gbm.predict(X_test)
predictions = predictions.as_data_frame()

import pandas as pd
y_test = pd.read_csv("y_test_MNIST.csv")
y_test = y_test.label

y_pred = pd.DataFrame(data = predictions)
y_pred.to_csv("y_pred_MNIST_H2O.csv")