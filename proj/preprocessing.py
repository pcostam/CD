# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:42:23 2018

@author: Margarida Costa
"""
import pandas as pd
# loading libraries
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

#Returns the data file. The class cls is obligatory for the method meanByClass	
def treatMissingValues(df, method, cl=None):
    if method == "constant":
        #1. use a global constant to fill in for missing values
        df = df.fillna(-1)
    elif method == "mean":
        #2. use attribute mean (or median if its discrete) for that column with na
        df = df.fillna(df.mean())
    elif method == "meanByClass":
        #3. use attribute mean (or median if its discrete) for the rows with certain class of the column with na
        for el in df[cl].unique():
            for col in df.columns[df.isna().any()].tolist():
                 df = df.fillna(df.loc[el,col].mean())
   
            
    elif method == "clustering":
        #todo
        pass
    elif method == "drop":   
        #6. Dropping axis labels with missing data
        df = df.dropna(axis=1)
    elif method == "interpolation": 
        #5. interpolation
        df = df.interpolate()
    elif method == "median":
        #6. use attribute median for that column with na
        df = df.fillna(df.median())
    elif method == "medianByClass":
        #7. use attribute median for the rows with certain class of the column with na
        for el in df[cl].unique():
            for col in df.columns[df.isna().any()].tolist():
                 df = df.fillna(df.loc[el,col].median())
                 
    return df

#Use for symbolic attributes with just 2 possible values. Doesn't create new columns. Returns de data file.
def treatSymbolicBinaryAtts(df, att, positiveClass):
    print("att", att)
    df[att] = np.array([ 1 if v == positiveClass else 0 for v in df[att] ])
    return df

#Use for symbolic with more than 2 possible values. Creates new columns with binary values for each symbolic class. Returns the data file.
def treatSymbolicAtt(df):
    label_encoder = LabelEncoder()
    dummy_encoder = OneHotEncoder()
    pdf = pd.DataFrame()
    for att in df.columns:
        if df[att].dtype == np.float64 or df[att].dtype == np.int64:
            pdf = pd.concat([pdf, df[att]], axis=1)
        else:
            if len(df[att].unique()) != 2:
                df[att] = label_encoder.fit_transform(df[att])
                # Fitting One Hot Encoding on train data
                temp = dummy_encoder.fit_transform(df[att].values.reshape(-1,1)).toarray()
                # Changing encoded features into a dataframe with new column names
                temp = pd.DataFrame(temp,
                                    columns=[(att + "_" + str(i)) for i in df[att].value_counts().index])
                # In side by side concatenation index values should be same
                # Setting the index values similar to the data frame
                temp = temp.set_index(df.index.values)
                # adding the new One Hot Encoded varibales to the dataframe
                pdf = pd.concat([pdf, temp], axis=1)
         
                    
        
    return pdf
	
#Returns the new X and Y for the already balanced data.
def treatUnbalancedData(df, method):
    X_train = df.drop( 'class', axis=1 ).values
    y_train = df[ 'class' ].values
    X_new = 0
    y_new = 0
    if(method == "SMOTE"):
        sm = SMOTE(random_state=12, ratio = 1.0)
        X_new, y_new = sm.fit_sample(X_train, y_train)
        #plot_2d_space(X_new, y_new, 'SMOTE')
        
    #random oversampling
    elif(method == "oversampling"):
        ros = RandomOverSampler()
        X_new, y_ros = ros.fit_sample(X_train, y_train)
        print(X_new.shape[0] - X_new.shape[0], 'new random picked points')
        #plot_2d_space(X_new, y_new, 'Random over-sampling')
    elif(method == "undersampling"):
        #random undersampling
        rus = RandomUnderSampler(return_indices=True)
        X_new, y_new, id_rus = rus.fit_sample(X_train, y_train)
        print('Removed indexes:', id_rus)
        #plot_2d_space(X_new, y_new, 'Random under-sampling')

    return X_new, y_new
