# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:42:50 2018

@author: anama
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:56:18 2018

@author: Margarida Costa
"""

import pandas as pd
from sklearn import cluster, mixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, mean_squared_error, completeness_score, mutual_info_score, homogeneity_score, v_measure_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import preprocessing as pp
from sklearn.feature_selection import chi2
#https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
print( '-----------------------------------' )

print( '-----------------------------------' )

data = {#pd.read_csv( f'{DATA_PATH}/{data_file}' )
    'test': pd.read_csv( r'.\data\Truck\aps_failure_test_set.csv', na_values="na"),
    'train': pd.read_csv( r'.\data\Truck\aps_failure_training_set.csv', na_values="na" ),
}

print( '>>> Loaded truck\'s data!' )
data['train'] = pp.treatSymbolicBinaryAtts(data['train'], "class", "pos")
data['train'] = pp.treatMissingValues(data['train'], "mean", "class")
X = data['train'].drop( 'class', axis=1 ).values
y = data['train']['class'].values
#center and scale data
X = StandardScaler().fit_transform(X)

  

#plot heatmaps to explore

#distance matrices
#euclidean distance
#pearson correlation
#feature selection - numerical data correlation or covariance
"""
from scipy.stats import pearsonr 

for j in range(X.shape[1]):
    pearsonr_coefficient, p_value = pearsonr(X[:,j],y)
"""

"""
K-means works by defining spherical clusters that are separable in a way so that the mean value converges 
towards the cluster center. Because of this, K-Means may underperform sometimes.
"""
results = []
algorithms = {}

#http://www.sthda.com/english/wiki/print.php?id=239
# using sinhouete https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
#http://ros-developer.com/2017/12/04/silhouette-coefficient-finding-optimal-number-clusters/
range_n_clusters = range(2,5)
k_clusters = 2

kmeans = cluster.KMeans(n_clusters=k_clusters, n_init=200)
kmeans.fit(X)



two_means = cluster.MiniBatchKMeans(n_clusters=k_clusters)

gmm = mixture.GaussianMixture(n_components=k_clusters, covariance_type='full')



"""
to assess the
uncertainty by calculating cluster p-values via multiscale
bootstrap resampling

"""

    
    
"""
In the absence of class information using the silhouette coefficient
b. In the presence of class information using:
    Silhouette analysis can be used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually
+1 far away, 0 very close
"""

algorithms =  (
                
                ('gmm', gmm),
                ('two_means', two_means),
                        )


xticklabels = []
all_ars = []
all_cs = []
MIS = []
H = []
VM = []
     
for name, algorithm in algorithms:  
    print(name)
    xticklabels.append(name)
    #adjusted Rand index
    y_labels = algorithm.fit_predict(X)
    all_ars.append(adjusted_rand_score(y, y_labels))
    #completeness
    all_cs.append(completeness_score(y, y_labels))
    #mutual Information based scores 
    MIS.append(mutual_info_score(y, y_labels ))
    #homogeneity )
    H.append(homogeneity_score(y, y_labels))
    #V-measure
    VM.append(v_measure_score(y, y_labels))
  
print(all_ars)
print(all_cs)
print(MIS)
print(H)
print(VM)
