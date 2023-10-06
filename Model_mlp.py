# %% Imports
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
from numpy import savetxt
from sklearn.metrics import confusion_matrix
import numpy as np

# %% Load dataset and create train-test sets
data = load_wine()
X = data.data
y = data.target
var_names = data.feature_names
var_names = [var_names[i][0:-5] for i in range(0, len(var_names))]
var_names = [var_names[i].title().replace(' ','') for i in range(0, len(var_names))]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Train model
regr = MLPClassifier(hidden_layer_sizes=(150,70),random_state=42, max_iter=500)
regr.fit(X_train, y_train)

# %% Get model predictions
y_pred = regr.predict(X_test)

# %% Save txt file
savetxt("predicted_classes.csv", y_pred, delimiter=",")
# %% Compute classification metrics
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa Score: {:.3f}".format(kappa))

cm = confusion_matrix(y_test, y_pred)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)
f1 = 2 * (precision * recall) / (precision + recall)
#%%
print("F1 Score: {:.3f} ".format(np.mean(f1)))
print("Recall: {:.3f}".format(np.mean(recall)))
print("Precision: {:.3f}".format(np.mean(precision)))