# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:31:30 2018

@author: anama
"""

import pandas as pd
      
# define column names
names = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class']

df = pd.read_csv("C:/Users/anama/Google Drive/MEIC/CD/labs/2/iris.csv", header=None, names=names)
print(df.head())

# loading libraries
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

def loadData():
    
    # create design matrix X and target vector y
    X = np.array(df.ix[1:, 0:4]) 	# end index is exclusive
    print("x shape:", X.shape)
    
    #print("x",  X)
    y = np.array(df['class']) 	# another way of indexing a pandas df
    y = np.delete(y, 0)
    print("y shape:", y.shape)
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=42, stratify=y)
    
    # loading library
    from sklearn.neighbors import KNeighborsClassifier
    
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    # instantiate learning model (k = 3)
    knn = KNeighborsClassifier(n_neighbors=3)
    
    # fitting the model
    model = knn.fit(X_train, y_train)
    
    # predict the response
    y_pred = model.predict(X_test)
    # evaluate accuracy
    print("accuracy", accuracy_score(y_test, y_pred))
    
    #labels = pd.unique(y)
    cm = confusion_matrix(y_test, y_pred, labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    print("report\n", classification_report(y_test, y_pred))
    print("confusion matrix", cm)
    
    
    # creating odd list of K for KNN
    myList = list(range(1,20))

    # subsetting just the odd ones
    neighbors = list(filter(lambda x: x % 2 != 0, myList))

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
        
    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]

    min_MSE = min(MSE)
    
    # determining best k
    optimal_k = neighbors[MSE.index(min_MSE)]
    print("The optimal number of neighbors is %d" % optimal_k)

    #instantiate the model
    gaussianNB = GaussianNB()
    
    gaussianNB.fit(X_train, y_train)
    print("accuracy GaussianNB", accuracy_score(y_test, y_pred))
    
    
    
    

