# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:19:48 2018

@author: Margarida Costa
"""

# -*- coding: utf-8 -*-
"""

"""
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
# loading libraries
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import preprocessing as pp
from sklearn.metrics import roc_curve, auc

print( '-----------------------------------' )

data = {#pd.read_csv( f'{DATA_PATH}/{data_file}' )
    'test': pd.read_csv( r'.\data\Truck\aps_failure_test_set.csv', na_values="na"),
    'train': pd.read_csv( r'.\data\Truck\aps_failure_training_set.csv', na_values="na" ),
}

print( '>>> Loaded truck\'s data!' )
data['test'] = pp.treatSymbolicBinaryAtts(data['test'], "class", "pos")
data['train'] = pp.treatSymbolicBinaryAtts(data['train'], "class", "pos")
data['train'] = pp.treatMissingValues(data['train'], "meanByClass", "class")
data['test'] = pp.treatMissingValues(data['test'], "mean")

#indentify outliers using IQR-score
data_o1 = data['train']
Q1 = data_o1.quantile(0.25)
Q3 = data_o1.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

#remove outliers
data_out = data_o1[~((data_o1 < (Q1 - 1.5 * IQR)) |(data_o1 > (Q3 + 1.5 * IQR))).any(axis=1)]
#data['train'] = data_out
#print(data['train']['class'].values)
#ao remover outliers passa a ter apenas 1 tipo de classe. Retirar????
X_test  = data['test'].drop( 'class', axis=1 ).values
y_test  = data['test'][ 'class' ].values
X_train, y_train =  pp.treatUnbalancedData(data['train'], "SMOTE")

# subsetting just the odd ones
neighbors = [1,3,5]

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    # fitting the model
    knnModel = knn.fit(X_train, y_train)
    # predict the response

    y_pred = knnModel.predict(X_test)
    print("scores", scores.mean())
    print("accuracy knn", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=pd.unique(y_train))
    print("confusion matrix", cm)
    print("TPrate knn ",cm[0][0]/(cm[0][0]+cm[1][0]))
    print("specificity knn", cm[1][1]/(cm[1][1]+cm[0][1]))


knn = KNeighborsClassifier(n_neighbors=1)
# fitting the model
knnModel = knn.fit(X_train, y_train)
   # calculate the fpr and tpr for all thresholds of the classification 
probs = knn.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



