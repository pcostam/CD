# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, make_scorer
import scikitplot as skplt
from preprocessing import treatSymbolicBinaryAtts, treatMissingValues
#pd.options.display.max_columns = None


# loading train and test dtaasets

train_data = pd.read_csv('data/Truck/aps_failure_training_set.csv', na_values='na')
test_data = pd.read_csv('data/Truck/aps_failure_test_set.csv', na_values='na')

print( '>>> Loaded truck\'s data!' )


# how many unique data points are in the training set
print(train_data['class'].value_counts())

train_data = treatSymbolicBinaryAtts(train_data, "class", "pos")
test_data = treatSymbolicBinaryAtts(test_data, "class", "pos")

train_data = treatMissingValues(train_data, "median", cl=None)
test_data = treatMissingValues(test_data, "median", cl=None)

X = train_data.drop('class', axis=1) 
y = train_data['class']

X_test_final = test_data.drop('class', axis=1)
y_test_final = test_data['class']

SEED = 1

# separating data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# applying random forest
trucks_rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1)

# compiling the model
trucks_rf_model.fit(X_train, y_train)
trucks_rf_model.score(X_test, y_test) #the score of the model

# predicting classes
y_pred = trucks_rf_model.predict(X_test)

# confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# visualizing the matrix
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
plt.show()