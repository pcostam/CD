# -*- coding: utf-8 -*-
"""

"""
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
# loading libraries
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

#from util.dummy_var import ProcessData

print( '-----------------------------------' )

DATA_PATH = r'./data'

data = {#pd.read_csv( f'{DATA_PATH}/{data_file}' )
    'test': pd.read_csv( r'.\data\Truck\aps_failure_test_set.csv' ),
    'train': pd.read_csv( r'.\data\Truck\aps_failure_training_set.csv' ),
}
print( '>>> Loaded truck\'s data!' )

#print( [*data] )

X_train = data['train'].drop( 'class', axis=1 ).values
X_test  = data['test'].drop( 'class', axis=1 ).values
y_train = data['train'][ 'class' ].values
y_test  = data['test'][ 'class' ].values

# Normalize data:
X_train[ X_train == 'na' ] = '-1'
X_test[ X_test == 'na' ] = '-1'
#y[ y == 'neg' ] = 0
#y[ y == 'pos' ] = 1

gnb = GaussianNB()
model = gnb.fit( X_train, y_train )

print( X_test )

y_pred = model.predict( X_test )

train_accuracy = gnb.score( X_train, y_train )

#cv = cross_val_score( naive_bayes, x_train, iris_y_train, cv=10 )

print( 'Accuracy:', train_accuracy )
print( 'y_pred:', y_pred )
#print( 'NB cross validation:', cv, sep='\n' )
print()