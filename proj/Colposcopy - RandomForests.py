# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, make_scorer, balanced_accuracy_score, auc, accuracy_score
import scikitplot as skplt
from preprocessing import treatSymbolicBinaryAtts, treatMissingValues
#pd.options.display.max_columns = None


# loading train and test dtaasets

green_data = pd.read_csv('data/Colposcopy/green.csv')
hinselmann_data = pd.read_csv('data/Colposcopy/hinselmann.csv')
schiller_data = pd.read_csv('data/Colposcopy/schiller.csv')

all_data = [green_data, hinselmann_data, schiller_data]

print( '>>> Loaded colposcopy data!' )

# how many unique data points are in the training set
print(green_data['consensus'].value_counts())
print(hinselmann_data['consensus'].value_counts())
print(schiller_data['consensus'].value_counts())

#no missing values

#no normalization needed for random forests
#X = []
#X_no_experts = []
#y = []


X = green_data.drop('consensus', axis=1)
X_no_experts = green_data.drop(['experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5', 'consensus'], axis=1)
y = green_data['consensus']

expert0 = green_data['experts::0'] 
expert1 = green_data['experts::1'] 
expert2 = green_data['experts::2'] 
expert3 = green_data['experts::3'] 
expert4 = green_data['experts::4'] 
expert5 = green_data['experts::5']
experts = [expert0, expert1, expert2, expert3, expert4, expert5]


SEED = 1

# separating data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)


#test random forest for diferent number of estimators
estimators = [100, 200, 500, 1000]
accs = []


for est in estimators:
    # applying random forest
    col_rf_model = RandomForestClassifier(n_estimators=est, class_weight='balanced', random_state=SEED, n_jobs=-1)
    
    # compiling the model
    col_rf_model.fit(X_train, y_train)
    
    # model accuracy
    print("Accuracy for ", est, " estimators: ", col_rf_model.score(X_test, y_test), "\n")
    
    # predicting classes
    y_pred = col_rf_model.predict(X_test)
    
    # model balanced acccuracy
    print("Balanced accuracy for ", est, "estimators: ", balanced_accuracy_score(y_pred, y_test), "\n")

#indeferent number of estimators, keep it at 200
col_rf_model = RandomForestClassifier(n_estimators=est, class_weight='balanced', random_state=SEED, n_jobs=-1)
col_rf_model.fit(X_train, y_train)

i = 0
for exp in experts:
    #we assume consensus at the better target thruth
    tn, fp, fn, tp = confusion_matrix(y, exp).ravel()
    print("scores for expert ", i, " :", tn, fp, fn, tp)
    
    #we create simplified costs (false predict = 1) to better evaluate experts predictions
    print("Cost of expert ", i, ": ", fp*1 + fn *1)
    
    #we use expert predictiions as predictions against the final consensus
    acc_score = accuracy_score(y, exp)
    print("scores for expert ", i, " :", acc_score, "\n")
    
    i += 1

# confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# visualizing the matrix
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
plt.show()


scores = col_rf_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

plt.title( 'ROC for Random Forests applied to Colposcopy' )
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print()
