print(" \n \n \n Begin imports \n \n \n")
import tpot
# check tpot version
print('tpot: %s' % tpot.__version__)
#Classification
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import csv 
import pandas as pd
from time import process_time

print("Begin Tpot")

df = pd.read_csv('/home/bench/notebooks/data/TomWilliams/Complete_undersampled.csv',sep = ',')

df.head()

train = pd.read_csv('/home/bench/notebooks/data/TomWilliams/train_IoT_Botnet.csv',sep = ',')
valid = pd.read_csv('/home/bench/notebooks/data/TomWilliams/test_IoT_Botnet.csv',sep = ',')

y_train = pd.read_csv('/home/bench/notebooks/data/TomWilliams/y_train_IoT_Botnet.csv',sep = ',')
X_train = pd.read_csv('/home/bench/notebooks/data/TomWilliams/x_train_IoT_Botnet.csv',sep = ',')
y_train = y_train.attack

tpotC = TPOTClassifier(generations=5,
    population_size=50,
    verbosity=2,
    scoring='roc_auc',
    random_state=42)

debut = process_time()
tpotC.fit(X_train, y_train)
fin = process_time()

print("Elapsed time in seconds : ",fin-debut)

X_test = pd.read_csv('/home/bench/notebooks/data/TomWilliams/x_test_IoT_Botnet.csv')
predictions = tpotC.predict(X_test)

y_test = pd.read_csv('/home/bench/notebooks/data/TomWilliams/y_test_IoT_Botnet.csv')

y_pred = pd.DataFrame(data = predictions)
y_pred.to_csv("y_pred_IoT_Botnet_Tpot.csv")

probs = tpotC.predict_proba(X_test)

probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)

fpr,tpr,_=roc_curve(y_test,probs)
roc_auc = auc(fpr,tpr)

plt.figure()
lw=2
plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve(area = %0.2f)' %roc_auc)
plt.plot([0,1],[0,1],color='navy',lw=lw, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print(confusion_matrix(y_test, predictions))