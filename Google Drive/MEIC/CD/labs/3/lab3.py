# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 18:02:19 2018

@author: anama
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from dummy_var import preprocessData
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve,  auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

  
bank_data = pd.read_csv( 'bank.csv' )
print("value_counts\n", bank_data['pep'].value_counts())
#NO 54,3%
#YES 45,6%
bank_dc = bank_data.select_dtypes(include=[object])
print( '>>> Loaded bank:', bank_data.shape )

#bank_data = preprocessData(bank_data)

print("head", bank_data.head())
print("fim")
bank_pep0_x = bank_data.drop( 'pep', axis=1 )
print(type(bank_pep0_x))
#print('##########',bank_pep0_x, sep='\n')
bank_pep0_x = preprocessData(bank_pep0_x).values
print("bank pep 0 ", bank_pep0_x)
bank_pep0_y = bank_data[ 'pep' ].values
#for v in bank_pep0_x:
print( type( bank_pep0_y ) )    
bank_pep0_y  = np.array([ 1 if v == 'YES' else 0 for v in bank_pep0_y ])
print("######## bank_pep0_y", bank_pep0_y)
print("shape bank_pep0_y", bank_pep0_y.shape)

bank_x_pep0_train, bank_x_pep0_test, bank_y_pep0_train, bank_y_pep0_test = train_test_split(bank_pep0_x, bank_pep0_y, test_size=0.70, random_state=42, stratify=bank_pep0_y)

#bank_x_pep1_train, bank_x_pep1_test, bank_y_pep1_train, bank_y_pep1_test = train_test_split(bank_pep1_x, bank_pep1_y, test_size=0.70, random_state=42, stratify=bank_pep1_y)

print( '>>>>>> NB:' )
naive_bayes = GaussianNB()
model = naive_bayes.fit( bank_x_pep0_train, bank_y_pep0_train )
print("bank_x_pep0_test", bank_x_pep0_test)
predY = model.predict(bank_x_pep0_test)

print('>>>>1. a')
print('accuracy pep 0', naive_bayes.score( bank_x_pep0_train, bank_y_pep0_train ))



print('>>>>1. b')

cm = confusion_matrix(bank_y_pep0_test, predY, labels=pd.unique(bank_pep0_y))
print("confusion matrix", cm)

print("TPrate",cm[0][0]/cm[0][0])
print("specificity", cm[1][1]/cm[1][1])


print('>>>>1. c')

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 0
print("model shape", predY.shape)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(bank_y_pep0_test[:, i], predY[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["pep"], tpr["pep"], _ = roc_curve(bank_y_pep0_test.ravel(), predY.ravel())
roc_auc["pep"] = auc(fpr["pep"], tpr["pep"])

print("fpr", fpr)
print("tpr", tpr)
print("roc_auc", roc_auc)
# Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr["pep"], tpr["pep"], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["pep"])
plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

print(">>>2a")
knn = KNeighborsClassifier(n_neighbors=3)
    
# fitting the model
knnModel = knn.fit(bank_x_pep0_train, bank_y_pep0_train)
    
# predict the response
y_pred = knnModel.predict(bank_x_pep0_test)

print("accuracy knn", accuracy_score(bank_y_pep0_test, y_pred))
    
print(">>>2b")
cm = confusion_matrix(bank_y_pep0_test, y_pred, labels=pd.unique(bank_pep0_y))
print("confusion matrix", cm)

print("TPrate knn ",cm[0][0]/(cm[0][0]+cm[1][0]))
print("specificity knn", cm[1][1]/(cm[1][1]+cm[0][1]))
print(">>>3 a.")
unbalanced_data = pd.read_csv( 'unbalanced.csv' )
print("value_counts\n", unbalanced_data['Outcome'].value_counts())
unbalanced_data = preprocessData(unbalanced_data)

unbalanced_x = unbalanced_data.drop( 'Outcome_0', axis=1 ).values
unbalanced_y = unbalanced_data[ 'Outcome_0' ].values

unbalanced_x_train, unbalanced_x_test, unbalanced_y_train, unbalanced_y_test = train_test_split(unbalanced_x, unbalanced_y, test_size=0.70, random_state=42, stratify=unbalanced_y)

   
print( '>>>>>> NB:' )
naive_bayes = GaussianNB()
model = naive_bayes.fit( unbalanced_x_train, unbalanced_y_train)
predY = model.predict( unbalanced_x_test)

print('>>>>3. a')
print('accuracy NB', naive_bayes.score( unbalanced_x_train, unbalanced_y_train ))
print('accuracy 3 KNN')

cm = confusion_matrix(unbalanced_y_test, predY, labels=pd.unique(unbalanced_y))
print("confusion matrix", cm)

print("TPrate",cm[0][0]/(cm[0][0]+cm[1][0]))
print("specificity", cm[1][1]/(cm[1][1]+cm[0][1]))
fpr, tpr, _ = roc_curve(unbalanced_y_test.ravel(), predY.ravel())
print("fpr" ,fpr,"tpr", tpr)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic 2')
plt.subplot(2, 1, 2)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


print(">>>>4")


#sm = SMOTE(random_state=12, ratio = 1.0)
#x_train_res, y_train_res = sm.fit_sample(x_train, y_train)