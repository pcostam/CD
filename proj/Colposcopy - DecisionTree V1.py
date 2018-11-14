# -*- coding: utf-8 -*-
"""

"""
# loading libraries
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
try:
    from sklearn.model_selection import train_test_split
except ImportError as ie:
    from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def run():
    print( '>>> Apply Decision Tree (entropy)' )
    dtc = DecisionTreeClassifier( criterion = 'entropy', random_state = 0 )
    model = dtc.fit( X_train, y_train )
    
    y_pred = model.predict( X_test )
    
    train_accuracy = dtc.score( X_train, y_train )
    cm = confusion_matrix( y_test, y_pred )
    sensitivity = cm[0][0] / ( cm[0][0] + cm[1][0] )
    specifity = cm[1][1] / ( cm[1][1] + cm[0][1] )
    
    print( 'Accuracy:', train_accuracy )
    print( 'Accuracy score:', accuracy_score( y_test, y_pred ) )
    #print( 'y_pred:', y_pred )
    print( "Sensitivity:", sensitivity )
    print( "Specificity:", specifity )
    
    # ----
    
    print( '>>> Apply Decision Tree (gini)' )
    dtc = DecisionTreeClassifier( criterion = 'gini', random_state = 0 )
    model = dtc.fit( X_train, y_train )
    
    y_pred = model.predict( X_test )
    
    train_accuracy = dtc.score( X_train, y_train )
    cm = confusion_matrix( y_test, y_pred )
    sensitivity = cm[0][0] / ( cm[0][0] + cm[1][0] )
    specifity = cm[1][1] / ( cm[1][1] + cm[0][1] )
    
    print( 'Accuracy:', train_accuracy )
    print( 'Accuracy score:', accuracy_score( y_test, y_pred ) )
    #print( 'y_pred:', y_pred )
    print( "Sensitivity:", sensitivity )
    print( "Specificity:", specifity )
    print()

for file_name in os.listdir( r'data\Colposcopy' ):
    
    print( '-----------------------------------' )
    print( '-----------------------------------' )

    data = pd.read_csv(
        os.path.join( '.', 'data', 'Colposcopy', file_name ),
        na_values = 'na'
    )
    
    print( '>>> Loaded colposcopy', file_name, 'data!' )

    # Normalize data:
    print( '--------------------------------------------' )
    print( 'Normalize data with a constant value of -1.' )
    X = data.fillna( -1 ).drop( 'consensus', axis = 1 )
    y = data.fillna( -1 )[ 'consensus' ].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.7,
        random_state = 42,
        stratify = y
    )
    run()
    
    print( '--------------------------------------------' )
    print( 'Normalize data with mean value.' )
    X = data.fillna( data.mean() ).drop( 'consensus', axis = 1 )
    y = data.fillna( data.mean() )[ 'consensus' ].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.7,
        random_state = 42,
        stratify = y
    )
    run()
    
    print( '--------------------------------------------' )
    print( 'Normalize data by dropping n/a.' )
    X = data.dropna( axis = 1 ).drop( 'consensus', axis = 1 )
    y = data.dropna( axis = 1 )[ 'consensus' ].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.7,
        random_state = 42,
        stratify = y
    )
    run()
    
    
    print( '--------------------------------------------' )
    print( 'Normalize data using interpolation.' )
    X = data.interpolate().drop( 'consensus', axis = 1 )
    y = data.interpolate()[ 'consensus' ].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.7,
        random_state = 42,
        stratify = y
    )
    run()