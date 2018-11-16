# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, make_scorer, balanced_accuracy_score, auc, accuracy_score
import scikitplot as skplt
from preprocessing import treatSymbolicBinaryAtts, treatMissingValues
#pd.options.display.max_columns = None


# loading train and test dtaasets

green_data = pd.read_csv('data/Colposcopy/green.csv')
hinselmann_data = pd.read_csv('data/Colposcopy/hinselmann.csv')
schiller_data = pd.read_csv('data/Colposcopy/schiller.csv')

all_data = [green_data, hinselmann_data, schiller_data]

print( '>>> Loaded colposcopy data!' )

# how many unique data points are in the training set
print(green_data['consensus'].value_counts())
print(hinselmann_data['consensus'].value_counts())
print(schiller_data['consensus'].value_counts())

#no missing values

#no normalization needed for random forests

def run(data, file_name):

    #X = data.drop('consensus', axis=1)
    X_no_experts = data.drop(['experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5', 'consensus'], axis=1)
    y = data['consensus']
    
    expert0 = data['experts::0'] 
    expert1 = data['experts::1'] 
    expert2 = data['experts::2'] 
    expert3 = data['experts::3'] 
    expert4 = data['experts::4'] 
    expert5 = data['experts::5']
    experts = [expert0, expert1, expert2, expert3, expert4, expert5]
    
    
    SEED = 1
    
    # separating data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_no_experts, y, test_size=0.3, stratify=y, random_state=SEED)
    
    
    #test random forest for diferent number of estimators
    estimators = [100, 200, 500, 1000]
    accs = []
    
    
    for est in estimators:
        # applying random forest
        col_rf_model = RandomForestClassifier(n_estimators=est, class_weight='balanced', random_state=SEED, n_jobs=-1)
        
        # compiling the model
        col_rf_model.fit(X_train, y_train)
        
        train_acc = col_rf_model.score(X_test, y_test)
        
        # model accuracy
        print("Accuracy for ", est, " estimators: ", train_acc, "\n")
        
        accs.append(train_acc)
        
        # predicting classes
        y_pred = col_rf_model.predict(X_test)
        
        # model balanced acccuracy
        print("Balanced accuracy for ", est, "estimators: ", balanced_accuracy_score(y_test, y_pred), "\n")
    
    #indeferent number of estimators, keep it at 200
    col_rf_model = RandomForestClassifier(n_estimators=est, class_weight='balanced', random_state=SEED, n_jobs=-1)
    col_rf_model.fit(X_train, y_train)
    y_pred = col_rf_model.predict(X_test)
    
    exp_costs = []
    exp_accs = []
    
    i = 0
    for exp in experts:
        #we assume consensus at the better target thruth
        tn, fp, fn, tp = confusion_matrix(y, exp).ravel()
        print("scores for expert ", i, " :", tn, fp, fn, tp)
        
        exp_cost = fp*1 + fn *1
        #we create simplified costs (false predict = 1) to better evaluate experts predictions
        print("Cost of expert ", i, ": ", exp_cost)
        exp_costs.append(exp_cost)
        
        #we use expert predictiions as predictions against the final consensus
        exp_acc = accuracy_score(y, exp)
        print("scores for expert ", i, " :", exp_acc, "\n")
        exp_accs.append(exp_acc)
        i += 1
    
    print("Mean of experts costs: ", np.mean(exp_costs))
    print("Mean of experts accuracies: ", np.mean(exp_accs))
    
    fig, ax1 = plt.subplots()
    exps = np.arange(0, 6)
    ax1.plot(exps, exp_costs, 'b-')
    ax1.set_xlabel('Experts')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Cost', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(exps, exp_accs, 'r-')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params('y', colors='r')
    
    fig.tight_layout()
    plt.title( 'Experts predictions on '+ file_name + ' colposcopy' )
    plt.show()
    
    # confusion matrix
    #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # visualizing the matrix
    #skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
    #plt.show()
    
    
    scores = col_rf_model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.title( 'ROC for Random Forests applied to '+ file_name + ' Colposcopy' )
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print()
    
    
for file_name in os.listdir( r'data\Colposcopy' ):

    data = pd.read_csv( os.path.join( '.', 'data', 'Colposcopy', file_name ) )
    
    print( '>>> Loaded colposcopy', file_name, 'data!' )
    
    run(data, file_name)