# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 22:43:08 2018

@author: anama
"""

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
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import preprocessing as pp
from sklearn.metrics import roc_curve, auc

print( '-----------------------------------' )

data = pd.read_csv( r'.\data\Colposcopy\green.csv', na_values="na")

print("fim")
X = data.drop('consensus', axis=1 )
print("data", data.head())
y = data['consensus'].values
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

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


knn = KNeighborsClassifier(n_neighbors=5)
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



