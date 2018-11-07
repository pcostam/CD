# -*- coding: utf-8 -*-
"""

"""

import os
from scipy.io import arff
import pandas as pd
import numpy as np
from IPython.display import display, HTML
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules

files = dict()

for file_name in os.listdir( os.path.join( '.', 'data' ) ):
    files[ file_name.strip( '.arff' ) ] = arff.loadarff(  os.path.join( '.', 'data', file_name ) )


print( '>>> running for marketing only, for now' )
marketingdata = files[ 'vote' ]

print( '1.1 -------------------------' )
df = pd.DataFrame( marketingdata[ 0 ])
display( HTML( df.to_html() ) )

with pd.option_context( 'display.max_rows', 10, 'display.max_columns', 14 ):
    print(df)
    
print( '1.2 -------------------------' )
df.fillna( '6', inplace = True ) #missings as new value
df = df()
for col in list( df ):
    attrs = []
    values = df[ col ].unique().tolist()
    values.sort()
    
    for value in values:
        attrs.append( f'{col}:{value}' )
        
    lb = MultiLabelBinarizer().fit_transform( df[ col ] )
    
    if( len( attrs ) == 2 ):
        v = list( map ( lambda x: 1 - x, lb ) )
        lb = np.concatenate( ( lb, v ), 1 )
    
    df2 = pd.DataFrame( data = lb, columns = attrs )
    
    if '6' in values:
        df2 = df2.drop( [ f'{col}:6' ], axis = 1 )
    
    df = df.drop( [ col ], axis = 1 )
    df = pd.concat( [ df, df2 ], axis = 1, join = 'inner' )

with pd.option_context( 'display.max_rows', 10, 'display.max_columns', 10 ):
    print( df )