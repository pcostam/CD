# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:23:53 2018

@author: anama
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import preprocessing as pp

def main():
    datatrucks = {#pd.read_csv( f'{DATA_PATH}/{data_file}' )
    'test': pd.read_csv( r'.\data\Truck\aps_failure_test_set.csv', na_values="na"),
    'train': pd.read_csv( r'.\data\Truck\aps_failure_training_set.csv', na_values="na" ),
    }
    print( '>>> Loaded truck\'s data!' )
    
    datacolposcopy = [
            pd.read_csv( r'.\data\Colposcopy\green.csv', na_values="na"),
            pd.read_csv( r'.\data\Colposcopy\hinselmann.csv', na_values="na"),
            pd.read_csv( r'.\data\Colposcopy\schiller.csv', na_values="na")
     ]
    print( '>>> Loaded colposcopy\'s data!' )
    
    for data in datacolposcopy:
        #preprocess colposcopy data
        X, y =  pp.treatUnbalancedData(data, "SMOTE")
        #find axis colposcopy data
        X = data.drop(['consensus', 'experts::0', 'experts::1','experts::2' ,'experts::3','experts::4','experts::5'], axis=1 ).values
        y = data['consensus'].values 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        KNN2.KNN_Colposcopy()
    
    
    
    #preprocess trucks data
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


    #axis truck
    X_test  = datatrucks['test'].drop( 'class', axis=1 ).values
    y_test  = datatrucks['test'][ 'class' ].values
    X_train, y_train =  pp.treatUnbalancedData(datatrucks['train'], "SMOTE")


