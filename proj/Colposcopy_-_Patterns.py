# -*- coding: utf-8 -*-
"""


"""
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from IPython.display import display, HTML
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt



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

S = [0.15, 0.30, 0.50, 0.75, 0.90, 0.95]
mean_lifts = []
nr_rules = []
for s in S:
    print( 'Support:', s )
    frequent_itemsets = apriori(df,min_support=s,use_colnames=True,max_len=2)
   
    res = list(frequent_itemsets['support'])
    print(min(res), max(res))
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.99)
    #for r in rules:
    #    print(rules[r])
        #print(r['antecedents'], "->", r['consequents'])
    #print( rules['antecedents'])
    for index, row in enumerate( rules['antecedents'] ):
        print( row, '->', rules['consequents'][index] )
        
    nr_rules.append(len(nr_rules))
    """print( 'RULES:')
    for rule in rules:
        #print( rules[rule] )
        print( rule, ':', sep='' )
        for k in rules[ rule ]:
            print(k)"""
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
    
  