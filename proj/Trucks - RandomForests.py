# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, make_scorer, balanced_accuracy_score, auc
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

train_data = treatMissingValues(train_data, "meanByClass", cl="class")
test_data = treatMissingValues(test_data, "mean", cl=None)

#no normalization needed in random forests

X = train_data.drop('class', axis=1) 
y = train_data['class']

X_test_final = test_data.drop('class', axis=1)
y_test_final = test_data['class']

SEED = 1

# separating data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

#test random forest for diferent number of estimators
estimators = [100, 200, 500, 1000]
accs = []
train_costs = []

for est in estimators:

    # applying random forest
    trucks_rf_model = RandomForestClassifier(n_estimators=est, class_weight='balanced', random_state=SEED, n_jobs=-1)
    
    # train the classifier
    trucks_rf_model.fit(X_train, y_train)
    
    train_acc = trucks_rf_model.score(X_test, y_test)
    
    # model accuracy
    print("Accuracy for ", est, " estimators: ", train_acc, "\n")
    
    accs.append(train_acc)

    # predicting classes
    y_pred = trucks_rf_model.predict(X_test)
    
    # model balanced acccuracy
    print("Balanced accuracy: ", balanced_accuracy_score(y_pred, y_test), "\n")
    
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    train_cost = fp*10 + fn *500

    print("Cost of the model for ", est, " estimators: ", train_cost, "\n")
    
    train_costs.append(train_cost)
    
plt.title( 'Trucks cost fuction according to different estimators' )
plt.plot(estimators, train_costs, 'b')
plt.xlim([0, 1000])
plt.ylabel('Cost of prediction model')
plt.xlabel('Nr. Estimators')
plt.show()
print()

#train the model for the best number of estimators
trucks_rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=-1)
trucks_rf_model.fit(X_train, y_train)

# visualizing the matrix
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
plt.show()

scores = trucks_rf_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

plt.title( 'ROC for Random Forests applied to Trucks' )
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print()

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
print("Final test accuracy: ", trucks_rf_model.score(X_test_final, y_test_final), "\n")

y_pred_test_final = trucks_rf_model.predict_proba(X_test_final)[:,1] > best_threshold

print("Final test balanced accuracy: ", balanced_accuracy_score(y_pred_test_final, y_test_final), "\n")

tn, fp, fn, tp = confusion_matrix(y_test_final, y_pred_test_final).ravel()
print("Cost of final test model: ", fp*10 + fn *500, "\n")