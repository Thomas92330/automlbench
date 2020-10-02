print(" \n \n \n Begin imports \n \n \n")
import tpot
# check tpot version
print('tpot: %s' % tpot.__version__)
#Classification
from tpot import TPOTClassifier
from sklearn.metrics import make_scorer
import pandas as pd
from time import process_time
# Any results you write to the current directory are saved as output.
import timeit 

print("Begin Tpot")

X_train = pd.read_csv('x_train_MNIST.csv',nrows = 20000)
y_train = pd.read_csv('y_train_MNIST.csv',nrows = 20000)
y_train = y_train.label

X_test = pd.read_csv('x_test_MNIST.csv',nrows = 6000)
y_test = pd.read_csv('y_test_MNIST.csv',nrows = 6000)
y_test = y_test.label

def my_custom_accuracy(y_true, y_pred):
    return float(sum(y_pred == y_true)) / len(y_true)

# Make a custom a scorer from the custom metric function
# Note: greater_is_better=False in make_scorer below would mean that the scoring function should be minimized.
my_custom_scorer = make_scorer(my_custom_accuracy, greater_is_better=True)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2,
                      scoring=my_custom_scorer)
tpot.fit(X_train, y_train)

X_test = pd.read_csv('/home/bench/notebooks/data/mnist/x_test_MNIST.csv')
predictions = tpot.predict(X_test)
y_pred = pd.DataFrame(data = predictions)
y_pred.to_csv("y_pred_MNIST_TPOT.csv")

