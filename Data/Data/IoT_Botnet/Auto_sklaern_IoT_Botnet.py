print("Begin imports")
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import sklearn.multiclass

print("Begin sklearn")

X_train = pd.read_csv("./TomWilliams/x_train_IoT_Botnet.csv")
X_test = pd.read_csv("./TomWilliams/x_test_IoT_Botnet.csv")
y_train = pd.read_csv("./TomWilliams/y_train_IoT_Botnet.csv")
y_test = pd.read_csv("./TomWilliams/y_test_IoT_Botnet.csv")

y_train = y_train.attack

start = time.time()
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task = 3600,
    ml_memory_limit=10000,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5}
)

# fit() changes the data in place, but refit needs the original data. We
# therefore copy the data. In practice, one should reload the data
automl.fit(X_train.copy(), y_train.copy())

elapsed_time=(time.time()-start)

predictions = automl.predict(X_test)

y_pred = pd.DataFrame(data = predictions)
y_pred.to_csv("y_pred_IoT_Botnet_Auto_sklearn.csv")