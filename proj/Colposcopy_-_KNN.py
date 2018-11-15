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
import os
# loading libraries
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import preprocessing as pp
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def run():

    X = data.drop(['consensus', 'experts::0', 'experts::1','experts::2' ,'experts::3','experts::4','experts::5'], axis=1 ).values
    y = data['consensus'].values

 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # subsetting just the odd ones
    neighbors = [1,3,5,7,9,11,13,15,17,19]

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    Sp = []
    F = []
    Se = []
    Acc = []
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
        # fitting the model
        knnModel = knn.fit(X_train, y_train)
        # predict the response
        y_pred = knnModel.predict(X_test)
        print("scores", scores.mean())
        accuracy = accuracy_score(y_test, y_pred)
        Acc.append(accuracy)
        print("accuracy knn", accuracy)
        cm = confusion_matrix(y_test, y_pred, labels=pd.unique(y_train))
        print("confusion matrix", cm)
        sensitivity = (cm[0][0])/(cm[0][0]+cm[1][0])
        print("Sensitivity(TPrate) knn ", sensitivity)
        Se.append(sensitivity)
        specificity = (cm[1][1])/(cm[1][1]+cm[0][1])
        print("Specificity knn", specificity)
        Sp.append(specificity)
        print("FPrate", 1-specificity )
        F.append(1-specificity)

    plt.figure()
    plt.plot(neighbors,Acc)
    plt.plot(neighbors,Sp)
    plt.plot(neighbors, Se)
    plt.xticks(neighbors)
    plt.ylabel('Performance')
    plt.xlabel('Neighbors')
    plt.legend(['Accuracy', 'Specificity', 'Sensitivity'], loc='best')
    
    plt.show()
    
    
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knnModel = knn.fit(X_train, y_train)
    # calculate the fpr and tpr for all thresholds of the classification
    probs = knn.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # method I: plt
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
for file_name in os.listdir( r'data\Colposcopy' ):
    data = pd.read_csv(
        os.path.join( '.', 'data', 'Colposcopy', file_name ),
        na_values = 'na'
    )
    
    data_o1 = data
    Q1 = data_o1.quantile(0.25)
    Q3 = data_o1.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    print(">>>>>>>>>>>>",file_name)
    data_out = data_o1[~((data_o1 < (Q1 - 1.5 * IQR)) |(data_o1 > (Q3 + 1.5 * IQR))).any(axis=1)]
    data = data_out
    run()
        



