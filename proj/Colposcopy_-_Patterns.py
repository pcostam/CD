# -*- coding: utf-8 -*-
"""


"""
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from IPython.display import display, HTML
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MultiLabelBinarizer


df= pd.read_csv( r'.\data\Colposcopy\green.csv', na_values="na")   
for col in list(df) :
    df[col] = pd.cut(df[col],3,labels=['0','1','2'])
    attrs = []
    values = df[col].unique().tolist()
    values.sort()
    for val in values : attrs.append("%s:%s"%(col,val))
    lb = LabelBinarizer().fit_transform(df[col])
    if(len(attrs)==2) :
        v = list(map(lambda x: 1 - x, lb))
        lb = np.concatenate((lb,v),1)
    df2 = pd.DataFrame(data=lb, columns=attrs)
    df = df.drop(columns=[col])
    df = pd.concat([df,df2], axis=1, join='inner')
    with pd.option_context('display.max_rows', 10, 'display.max_columns', 8): print(df)

print("fim") 
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
display(HTML(rules.to_html()))
print(rules)
#3
print(rules['lift'])
print(rules['confidence'])
print(rules['support'])
print(rules['leverage'])
print(rules['conviction'])

#scipy.stats.chisquare(frequent_itemsets,f_exp=None,ddof=0,axis=0)

