# -*- coding: utf-8 -*-
"""

"""
# loading libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

#from preprocessing import treatMissingValues

print( '-----------------------------------' )


data = {
    'test': pd.read_csv( r'.\data\Truck\aps_failure_test_set.csv' ),
    'train': pd.read_csv( r'.\data\Truck\aps_failure_training_set.csv' ),
}
print( '>>> Loaded truck\'s data!' )

def run():
    print( '>>> Apply Decision Tree (entropy)' )
    dtc = DecisionTreeClassifier( criterion = 'entropy', random_state = 0 )
    model = dtc.fit( X_train, y_train )
    
    y_pred = model.predict( X_test )
    
    train_accuracy = dtc.score( X_train, y_train )
    print( 'Accuracy:', train_accuracy )
    print( 'y_pred:', y_pred )
    
    print( '>>> Apply Decision Tree (gini)' )
    dtc = DecisionTreeClassifier( criterion = 'gini', random_state = 0 )
    model = dtc.fit( X_train, y_train )
    
    y_pred = model.predict( X_test )
    
    train_accuracy = dtc.score( X_train, y_train )
    print( 'Accuracy:', train_accuracy )
    print( 'y_pred:', y_pred )
    print()

# Normalize data:
print( '--------------------------------------------' )
print( 'Normalize data with a constant value of -1.' )
X_train = data['train'].drop( 'class', axis=1 ).values
X_test  = data['test'].drop( 'class', axis=1 ).values
y_train = data['train'][ 'class' ].values
y_test  = data['test'][ 'class' ].values
X_train[ X_train == 'na' ] = '-1'
X_test[ X_test == 'na' ] = '-1'
run()

print( '--------------------------------------------' )
print( 'Normalize data with mean value.' )
data['train'].fillna( data['train'].mean() )
X_train = data['train'].drop( 'class', axis=1 ).values
X_test  = data['test'].drop( 'class', axis=1 ).values
y_train = data['train'][ 'class' ].values
y_test  = data['test'][ 'class' ].values
run()

print( '--------------------------------------------' )
print( 'Normalize data by dropping n/a.' )
data['train'].dropna( axis = 1 )
X_train = data['train'].drop( 'class', axis=1 ).values
X_test  = data['test'].drop( 'class', axis=1 ).values
y_train = data['train'][ 'class' ].values
y_test  = data['test'][ 'class' ].values
run()


print( '--------------------------------------------' )
print( 'Normalize data using interpolation.' )
data['train'].interpolate()
X_train = data['train'].drop( 'class', axis=1 ).values
X_test  = data['test'].drop( 'class', axis=1 ).values
y_train = data['train'][ 'class' ].values
y_test  = data['test'][ 'class' ].values
run()
