print("Begin imports")
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML
from time import process_time
print("Begin h2o")
h2o.init()
df = h2o.upload_file('./TomWilliams/train_IoT_Botnet.csv')
df["attack"] = df["attack"].asfactor()
response = "attack"
predictors=[]
for col in df.columns:
    if col != 'attack':
        predictors.append(col)

train = h2o.upload_file('./TomWilliams/train_IoT_Botnet.csv')
valid = h2o.upload_file('./TomWilliams/test_IoT_Botnet.csv')

IoT_Botnet_gbm = H2OGradientBoostingEstimator()

t1_start = process_time()
IoT_Botnet_gbm.train(x = predictors,
               y = response,
               training_frame = train,
               validation_frame = valid)
t1_stop = process_time()

print( IoT_Botnet_gbm)

print("Elapsed time in seconds : ",t1_stop-t1_start)  

X_test = h2o.upload_file('./TomWilliams/x_test_IoT_Botnet.csv')
predictions = IoT_Botnet_gbm.predict(X_test)
predictions = predictions.as_data_frame()

import pandas as pd

y_pred = pd.DataFrame(data = predictions)
y_pred.to_csv("y_pred_IoT_Botnet_H2O.csv")