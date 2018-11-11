# -*- coding: utf-8 -*-
"""

"""
# loading libraries
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
try:
    from sklearn.model_selection import train_test_split
except ImportError as ie:
    from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
        
def run( file_name ):
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
    fpr, tpr, threshold = roc_curve( y_test, y_pred )
    plt.title( f'ROC for {file_name}')
    plt.plot(fpr, tpr, 'b', label = 'AUC = ')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
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
    run( file_name )
    
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
    run( file_name )
    
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
    run( file_name )
    
    
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
    run( file_name )