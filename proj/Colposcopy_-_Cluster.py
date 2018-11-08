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
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, mean_squared_error
import numpy as np


print( '-----------------------------------' )



data = pd.read_csv( r'.\data\Colposcopy\green.csv', na_values="na")


X = data.drop(['consensus', 'experts::0', 'experts::1','experts::2' ,'experts::3','experts::4','experts::5'], axis=1 ).values
print("X", len(X))
y = data['consensus'].values
print("y", len(y), y)

#center and scale data
X = StandardScaler().fit_transform(X)


#plot heatmaps to explore

#distance matrices
#euclidean distance
#pearson correlation

"""
K-means works by defining spherical clusters that are separable in a way so that the mean value converges 
towards the cluster center. Because of this, K-Means may underperform sometimes.
"""
k_clusters = 2
results = []
algorithms = {}

"""
k-Means with parameterized distance metrics:
Kmeans sklearn uses euclidean measures
"""
algorithms['kmeans'] = cluster.KMeans(n_clusters=k_clusters, n_init=200)



"""
The main idea behind Agglomerative clustering is that each node first starts in its own cluster,
and then pairs of clusters recursively merge together in a way that minimally increases a given 
linkage distance.
The main advantage of Agglomerative clustering (and hierarchical clustering in general) 
is that you don’t need to specify the number of clusters.
"""

ward = cluster.AgglomerativeClustering(
n_clusters=k_clusters, linkage='ward')
complete = cluster.AgglomerativeClustering(
n_clusters=k_clusters, linkage='complete')
average = cluster.AgglomerativeClustering(
n_clusters=k_clusters, linkage='average')
single = cluster.AgglomerativeClustering(
n_clusters=k_clusters, linkage='single')


agglomerative_algorithms = (
                            ('ward', ward),
                            ('complete', complete),
                            ('average', average),
                            ('single', single)
                           )

#chisquare(X_train, y_train)
#chisquare(X_test, y_test)
    
for name, algorithm in agglomerative_algorithms:
   print(name)
   #clf.cluster_centers_
   y_labels = algorithm.fit_predict(X)
    #Labels of each point


    # !! Get the indices of the points for each corresponding cluster
   mydict = {i: np.where(y_labels == i)[0] for i in range(algorithm.n_clusters)}

   print(mydict)
    
   squared_errors = y - y_labels ** 2
   #np.sum(squared_errors)
   #chi square test

    


"""

The Spectral clustering technique applies clustering to a projection of the normalized Laplacian.

"""

algorithms['spectral'] = cluster.SpectralClustering(n_clusters=k_clusters, affinity="precomputed", n_init=200)
spectral = cluster.SpectralClustering(n_clusters=k_clusters, eigen_solver='arpack', affinity="nearest_neighbors")

"""Affinity Propagation
"""
algorithms['affinity'] = cluster.AffinityPropagation(damping=0.6, preference=-200)


"""
The algorithm takes small batches(randomly chosen) of the dataset for each iteration. It then assigns a cluster to each data point in the batch, depending on the previous locations of the cluster centroids. It then updates the locations of cluster centroids based on the new points from the batch. The update is a gradient descent update, which is significantly faster than a normal Batch K-Means update. 
"""


two_means = cluster.MiniBatchKMeans(n_clusters=k_clusters)


"""
DBSCAN
 In DBSCAN, there are no centroids, and clusters are formed by linking nearby points to one another.
 
"""

dbscan = cluster.DBSCAN(eps=0.3)

birch = cluster.Birch(n_clusters=k_clusters)

"""
 Gaussian Mixture
 attempts to find a mixture of multi-dimensional Gaussian probability distributions that best model any input dataset
"""
gmm = mixture.GaussianMixture(n_components=k_clusters, covariance_type='full')


bandwidth = cluster.estimate_bandwidth(X, quantile=0.2)
ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)


"""
to assess the
uncertainty by calculating cluster p-values via multiscale
bootstrap resampling

"""
    
#evalue
    
    
    
"""
In the absence of class information using the silhouette coefficient
b. In the presence of class information using:
    Silhouette analysis can be used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually
+1 far away, 0 very close
"""

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

     

"""
i. adjusted Rand index
"""
clusterer = cluster.KMeans(n_clusters=k_clusters, random_state=10)
y_labels = algorithm.fit_predict(X)

print("rand_index_score", adjusted_rand_score(y, y_labels))

"""
ii. sum of squared errors
"""
mean_squared_error(y, y_labels)


"""
iii. mutual Information based scores (in Python)
"""
"""
iv. homogeneity, completeness and V-measure (in Python)
"""
