# -*- coding: utf-8 -*-
"""

"""
# loading libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

#from util.dummy_var import ProcessData

print( '-----------------------------------' )


data = {
    'test': pd.read_csv( r'.\data\Truck\aps_failure_test_set.csv', na_values='na' ),
    'train': pd.read_csv( r'.\data\Truck\aps_failure_training_set.csv', na_values='na' ),
}
print( '>>> Loaded truck\'s data!' )

def run():
    print( '>>> Apply Naive Bayes' )
    gnb = GaussianNB()
    model = gnb.fit( X_train, y_train )
    
    y_pred = model.predict( X_test )
    
    train_accuracy = gnb.score( X_train, y_train )
    
    #cv = cross_val_score( naive_bayes, x_train, iris_y_train, cv=10 )
    
    print( 'Accuracy:', train_accuracy )
    print( 'Accuracy score:', accuracy_score( y_test, y_pred ) )
    print( 'y_pred:', y_pred )
    #print( 'NB cross validation:', cv, sep='\n' )
    print()


# Normalize data:
print( '--------------------------------------------' )
print( 'Normalize data with a constant value of -1.' )
data['train'] = data['train'].fillna( -1 )
data['test'] = data['test'].fillna( -1 )
X_train = data['train'].drop( 'class', axis=1 ).values
X_test  = data['test'].drop( 'class', axis=1 ).values
y_train = data['train'][ 'class' ].values
y_test  = data['test'][ 'class' ].values
#X_train[ X_train == 'na' ] = '-1'
#X_test[ X_test == 'na' ] = '-1'
run()

print( '--------------------------------------------' )
print( 'Normalize data with mean value.' )
data['train'] = data['train'].fillna( data['train'].mean() )
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