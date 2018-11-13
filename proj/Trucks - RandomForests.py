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
# model accuracy
print("Acuracy: ", trucks_rf_model.score(X_test, y_test), "\n")

# predicting classes
y_pred = trucks_rf_model.predict(X_test)

# confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# visualizing the matrix
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
plt.show()

print("Cost of the model: ", fp*10 + fn *500, "\n")


scores = trucks_rf_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, scores)

min_cost = np.inf
best_threshold = 0.5
costs = []

for threshold in thresholds:
    y_pred_threshold = scores > threshold
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
    cost = 10*fp + 500*fn
    costs.append(cost)
    if cost < min_cost:
        min_cost = cost
        best_threshold = threshold
        
print("Best threshold: {:.4f}".format(best_threshold))
print("Min cost: {:.2f}".format(min_cost))


# using the test dataset
print("Final test acuracy: ", trucks_rf_model.score(X_test_final, y_test_final), "\n")

y_pred_test_final = trucks_rf_model.predict_proba(X_test_final)[:,1] > best_threshold
tn, fp, fn, tp = confusion_matrix(y_test_final, y_pred_test_final).ravel()
print("Cost of final test model: ", fp*10 + fn *500, "\n")