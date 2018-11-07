
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
# loading libraries
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
plt.style.use('ggplot')


print( '-----------------------------------' )
iris_data = pd.read_csv( 'iris.csv' )
print( '>>> Loaded iris:', iris_data.shape )


iris_x = iris_data.drop( 'class', axis=1 ).values
iris_y = iris_data[ 'class' ].values

iris_x_train, iris_x_test, iris_y_train, iris_y_test = train_test_split(
    iris_x,
    iris_y,
    train_size=0.7,
    random_state=42
)
#######################################
knn = KNeighborsClassifier()
knn.fit( iris_x_train, iris_y_train)

train_accuracy = knn.score( iris_x_train, iris_y_train )

cv = cross_val_score( knn, iris_x_train, iris_y_train, cv=10 )

print( 'KNN Accuracy:', train_accuracy )
print( 'KNN cross validation:', cv, sep='\n' )
print()

####################
naive_bayes = GaussianNB()
naive_bayes.fit( iris_x_train, iris_y_train )

train_accuracy = naive_bayes.score( iris_x_train, iris_y_train )

cv = cross_val_score( naive_bayes, iris_x_train, iris_y_train, cv=10 )

print( 'NB Accuracy:', train_accuracy )
print( 'NB cross validation:', cv, sep='\n' )
print()


"""
##########################################
"""
glass_data = pd.read_csv( 'glass.csv' )
print( '-----------------------------------' )
print( '>>> Loaded glass:', glass_data.shape )

glass_x = glass_data.drop( 'Type', axis=1 ).values
glass_y = glass_data[ 'Type' ].values

glass_x_train, glass_x_test, glass_y_train, glass_y_test = train_test_split(
    glass_x,
    glass_y,
    train_size=0.7,
    random_state=42,
    stratify=glass_y
)

print( '>>>>>> KNN:' )
for i, k in enumerate( [ 1, 5, 10, 15, 50, 100 ] ):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier( n_neighbors=k )
    
    #Fit the model
    model = knn.fit(glass_x_train, glass_y_train)
    
    predY = model.predict(glass_x_test)
    #Compute accuracy on the training set
    accuracy_train = knn.score( glass_x_train, glass_y_train )
    accuracy_test = knn.score( glass_x_test, glass_y_test )
    
    #Cross validation
    #cv = cross_val_score( knn, glass_x_train, glass_y_train, cv=10 )
    
    print( 'Accuracy for', k, 'neighbours is', accuracy_train, 'vs.', accuracy_test )
    #print( 'Cross validation for', k, 'neighbours is', cv, sep='\n\t')

print()

print( '>>>>>> NB:' )
naive_bayes = GaussianNB()
model = naive_bayes.fit( glass_x_train, glass_y_train )
predY = model.predict(glass_x_test)

train_accuracy = naive_bayes.score( glass_x_train, glass_y_train )

cv = cross_val_score( naive_bayes, glass_x_train, glass_y_train, cv=10, scoring='accuracy' )


print( 'NB Accuracy:', train_accuracy )
print( 'NB cross validation:', cv, sep='\n\t' )
'''
plt.figure()
plt.plot(glass_x_test, predY, label="Model")
plt.plot(glass_x_test, model, 'x')
plt.scatter(glass_x, glass_y, label="Samples")'''
