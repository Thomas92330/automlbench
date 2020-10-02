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

X_train = pd.read_csv("x_train_MNIST.csv")
X_test = pd.read_csv("x_test_MNIST.csv")
y_train = pd.read_csv("y_train_MNIST.csv")
y_test = pd.read_csv("y_test_MNIST.csv")

y_train = y_train.label

start = time.time()
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task = 360,
    ml_memory_limit=10000
)

# fit() changes the data in place, but refit needs the original data. We
# therefore copy the data. In practice, one should reload the data
automl.fit(X_train.copy(), y_train.copy())

elapsed_time=(time.time()-start)

predictions = automl.predict(X_test)

y_pred = pd.DataFrame(data = predictions)
y_pred.to_csv("y_pred_MNIST_Auto_sklearn.csv")