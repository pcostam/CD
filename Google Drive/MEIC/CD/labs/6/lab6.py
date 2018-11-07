# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:06:25 2018

@author: anama
"""

import arff
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from IPython.display import display, HTML
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MultiLabelBinarizer


#1

"""
from scipy.io import arff
votedata = arff.loadarff('vote.arff')
df = pd.DataFrame(votedata[0])
print("df", df)
"""
data = arff.load(open('vote.arff'))
print(data)
attrs = []
for attr in data['attributes']: attrs.append(attr[0])
df = pd.DataFrame(data=data['data'], columns=attrs)
display(HTML(df.to_html()))
with pd.option_context('display.max_rows', 10, 'display.max_columns', 14): print(df)


df.fillna(0, inplace = True) #missings as new value
print(df['Class'])
df["Class"] = np.array([ 1 if v == "republican" else 0 for v in df["Class"] ])
df = df.replace("y", 1)
df = df.replace("n", 0)

print("df", df)
#2
#The Apriori algorithm tries to extract rules for each possible combination of items. 
"""
It does so using a "bottom up" approach, first identifying individual items that satisfy a minimum occurence threshold. 
It then extends the item set, adding one item at a time and checking 
if the resulting item set still satisfies the specified threshold.
 The algorithm stops when there are no more items to add that meet the minimum occurrence requirement.
"""
"""
Once the item sets have been generated using apriori, we can start mining association rules. 
Given that we are only looking at item sets of size 2, the association rules we will generate will be of the form {A} -> {B}. One common application of these rules is in the domain of recommender systems, where customers who purchased item A are recommended item B.
3  metrics when evaluating associations rules:
    1. support
    2. confidence
    3. lift
"""

    
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

import numpy as np
from pymining import seqmining
from prefixspan import PrefixSpan


fp = open("data/spm/sign.txt")
line = fp.readline()
seqdata = []
while line:
    seqdata.append(line.strip().split(' '))
    line = fp.readline()
fp.close()
freq_seqs = seqmining.freq_seq_enum(seqdata, 550)
sorted(freq_seqs)

ps = PrefixSpan(seqdata)
print(ps.frequent(500, closed=True),'\n')
