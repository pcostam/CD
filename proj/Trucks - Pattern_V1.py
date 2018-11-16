# -*- coding: utf-8 -*-
"""
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import preprocessing as pp
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE


df = pd.read_csv(
    os.path.join( '.', 'data', 'Truck', 'aps_failure_training_set.csv' ),
    na_values='na'
)
df = pp.treatSymbolicBinaryAtts(df, "class", "pos")
df = pp.treatMissingValues(df, "mean")
print(df.head())


for col in list(df):
    if col not in ['class'] :
        df[col] = pd.cut(df[col],1,labels=['0'])
    attrs = []
    values = df[col].unique().tolist()
    values.sort()
    for val in values : attrs.append("%s:%s"%(col,val))
    lb = LabelBinarizer().fit_transform(df[col])
    if(len(attrs)==2) :
        v = list(map(lambda x: 1 - x, lb))
        lb = np.concatenate((lb,v),1)
        
        
    df2 = pd.DataFrame(data=lb, columns=attrs)
    df = df.drop([col], axis=1)
    df = pd.concat([df,df2], axis=1, join='inner')    
 
    #with pd.option_context('display.max_rows', 10, 'display.max_columns', 8):
        #print(df)
   
#print('df', df)

S = [0.9833]
mean_lifts = []
nr_rules = []
for s in S:
    print( 'Support:', s )
    frequent_itemsets = apriori(df,min_support=s,use_colnames=True,max_len=2)
   
    res = list(frequent_itemsets['support'])
    print(min(res), max(res))
  
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.99)

    for index, row in enumerate( rules['antecedents'] ):
        print( row, '->', rules['consequents'][index] )
        
    nr_rules.append(len(nr_rules))

    print('RULES:' , rules)
    print(frequent_itemsets['itemsets'])
    #3
    lifts = [lift for lift in rules['lift']]
    print('lift for support', s, lifts)
    mean_lift = sum(lifts)/len(lifts)
    mean_lifts.append(mean_lift)
    

    
plt.figure()
    
plt.plot(S, mean_lifts)
plt.ylabel('Mean Lift')
plt.xlabel('support')
    
plt.show()    

plt.figure()
plt.plot(S, nr_rules)
plt.ylabel('Nr. rules')
plt.xlabel('support')
    
plt.show()  
    
  