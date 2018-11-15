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
import os
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

def run():
    data = pd.read_csv( r'.\data\Colposcopy\green.csv', na_values="na")
    
    #indentify outliers using IQR-score
    data_o1 = data
    Q1 = data_o1.quantile(0.25)
    Q3 = data_o1.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    
    #remove outliers
    data_out = data_o1[~((data_o1 < (Q1 - 1.5 * IQR)) |(data_o1 > (Q3 + 1.5 * IQR))).any(axis=1)]
    data = data_out
    
    X = data.drop(['consensus', 'experts::0', 'experts::1','experts::2' ,'experts::3','experts::4','experts::5'], axis=1 ).values
    y = data['consensus'].values
    
    #center and scale data
    X = StandardScaler().fit_transform(X)
    
      
    
    #plot heatmaps to explore
    
    #distance matrices
    #euclidean distance
    #pearson correlation
    #feature selection - numerical data correlation or covariance
    from scipy.stats import pearsonr 
    
    for j in range(X.shape[1]):
        pearsonr_coefficient, p_value = pearsonr(X[:,j],y)
    
    """
    K-means works by defining spherical clusters that are separable in a way so that the mean value converges 
    towards the cluster center. Because of this, K-Means may underperform sometimes.
    """
    k_clusters = 2
    results = []
    algorithms = {}
    
    #http://www.sthda.com/english/wiki/print.php?id=239
    # using sinhouete https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    #http://ros-developer.com/2017/12/04/silhouette-coefficient-finding-optimal-number-clusters/
    range_n_clusters = range(2,19)
    silhouette_score_values = []
    all_ars = []
    all_cs = []
    MIS = []
    H = []
    VM = []
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value
        clusterer = cluster.KMeans(n_clusters=n_clusters, n_init=200)
        cluster_labels = clusterer.fit_predict(X)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_score_values.append(silhouette_avg)
        print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
        all_ars.append(adjusted_rand_score(y, cluster_labels))
        print("rand index score", adjusted_rand_score(y, cluster_labels))
        #completeness
        print("complenetess score", completeness_score(y, cluster_labels))
        all_cs.append(completeness_score(y, cluster_labels))
        #mutual Information based scores 
        print("mutual information score", mutual_info_score(y, cluster_labels ))
        MIS.append(mutual_info_score(y, cluster_labels ))
        #homogeneity 
        print("homogeneity", homogeneity_score(y, cluster_labels))
        H.append(homogeneity_score(y, cluster_labels))
        #V-measure
        print("v-measure", v_measure_score(y, cluster_labels))
        VM.append(v_measure_score(y, cluster_labels))
    
    plt.figure()
    plt.plot(range_n_clusters, silhouette_score_values)
    plt.plot(range_n_clusters,all_ars)
    plt.plot(range_n_clusters, all_cs)
    plt.plot(range_n_clusters, MIS)
    plt.plot(range_n_clusters, H)
    plt.plot(range_n_clusters, VM)
    plt.xticks(range_n_clusters)
    plt.ylabel('Performance')
    plt.xlabel('N-Clusters')
    plt.legend(['Silhouette', 'Rand Index Score', 'Completeness Score','Mutual information score' ,'Homogeneity' , 'V-measure'], loc='best')
    plt.show()
        
        
    #k_clusters = range_n_clusters[silhouette_score_values.index(max(silhouette_score_values))]
    k_clusters = 8
    print("k_clusters", k_clusters)
    
    """
    k-Means with parameterized distance metrics:
    Kmeans sklearn uses euclidean measures
    """
    kmeans = cluster.KMeans(n_clusters=k_clusters, n_init=200)
    kmeans.fit(X)
    #save centroids for further mapping in agglomerative cluster
    Kx = kmeans.cluster_centers_
    print(len(kmeans.labels_))
    print("kx", len(Kx))
    Kx_mapping = {case:cluster for case,
     cluster in enumerate(kmeans.labels_)}
    
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
       #cluster of each point
       algorithm.fit(X)
       # Get the indices of the points for each corresponding cluster
       mydict = {i: np.where(algorithm.labels_ == i)[0] for i in range(algorithm.n_clusters)} 
       #what cluster centroids from kmeans were assigned to the agglomerative cluster
       H_mapping = {case:cluster for case,cluster in enumerate(algorithm.labels_)}
    
       final_mapping = {case:H_mapping[Kx_mapping[case]] for case in Kx_mapping}
    
       centroid_mapping = {}
       for c in final_mapping.values():
           try: 
               centroid_mapping[c] 
           except KeyError:
               centroid_mapping[c] = Kx[0]
    
       s = 0
       for c in centroid_mapping.values():
           for x in range(len(X)):
               s += (np.linalg.norm(c - x)) ** 2
        
       print("sum of squared errors", s)
       
       #plt.figure()
       #plt.title('Hierarchical Clustering Dendrogram')
       #plot_dendrogram(algorithm, labels=algorithm.labels_)
       #plt.show()
       
    
        
    
    
    
    """
    
    The Spectral clustering technique applies clustering to a projection of the normalized Laplacian.
    
    """
    
    spectral = cluster.SpectralClustering(n_clusters=k_clusters, eigen_solver='arpack', affinity="nearest_neighbors")
    
    """Affinity Propagation
    """
    silhouette_score_values = []
    all_ars = []
    all_cs = []
    MIS = []
    H = []
    VM = []
    print(silhouette_score_values)
    damping_values = [0.5,0.6,0.7,0.8,0.9]
    for value in damping_values:
        print(value)
        affinity = cluster.AffinityPropagation(damping=value)
        labels = affinity.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_score_values.append(silhouette_avg)
        all_ars.append(adjusted_rand_score(y, labels))
        all_cs.append(completeness_score(y, labels))
        MIS.append(mutual_info_score(y, labels ))
        #homogeneity 
        H.append(homogeneity_score(y, labels))
        #V-measure
        VM.append(v_measure_score(y, labels))
        
    plt.figure()
    plt.plot(damping_values,silhouette_score_values)
    plt.plot(damping_values,all_ars)
    plt.plot(damping_values, all_cs)
    plt.plot(damping_values, MIS)
    plt.plot(damping_values, H)
    plt.plot(damping_values, VM)
    plt.xticks(damping_values)
    plt.ylabel('Performance')
    plt.xlabel('Damping')
    plt.legend(['Silhouette', 'Rand Index Score', 'Completeness Score','Mutual information score' ,'Homogeneity' , 'V-measure'], loc='upper right')
    plt.show()
        
    print("max silhouette score", max(silhouette_score_values))
    
    best_damping = damping_values[silhouette_score_values.index(max(silhouette_score_values))]
    print("max_affinity", best_damping)
    
    affinity = cluster.AffinityPropagation(damping=best_damping,preference=-200)
    
    
    """
    The algorithm takes small batches(randomly chosen) of the dataset for each iteration. It then assigns a cluster to each data point in the batch, depending on the previous locations of the cluster centroids. It then updates the locations of cluster centroids based on the new points from the batch. The update is a gradient descent update, which is significantly faster than a normal Batch K-Means update. 
    """
    
    two_means = cluster.MiniBatchKMeans(n_clusters=k_clusters)
    
    
    """
    DBSCAN
     In DBSCAN, there are no centroids, and clusters are formed by linking nearby points to one another.
     
    """
    #http://www.sthda.com/english/wiki/print.php?id=246
    ns = 3
    nbrs = NearestNeighbors(n_neighbors=ns).fit(X)
    distances, indices = nbrs.kneighbors(X)
    print("distances", distances)
    print("indices", indices)
    distances = sorted(distances[:,-1])
    print(distances)
    points = list(range(1,len(indices)+1))
    plt.figure()
    plt.ylabel('Distance knn')
    plt.xlabel('Points sample sorted by distance')
    plt.plot(points, distances)
    plt.show()
    
    eps_range = range(2,15)
 
    _labels = []
    silhouette_score_values = []
    all_ars = []
    all_cs = []
    MIS = []
    H = []
    VM = []
    all_cluster = []
    for e in eps_range :
        dbscan = cluster.DBSCAN(eps=e)
        _labels = dbscan.fit_predict(X)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(_labels)) - (1 if -1 in labels else 0)
        all_cluster.append(n_clusters_)
        #silhouette_score(X, _labels)
        all_ars.append(adjusted_rand_score(y, _labels))
    
        all_cs.append(completeness_score(y, _labels))
    
        MIS.append(mutual_info_score(y, _labels ))
    
        H.append(homogeneity_score(y,_labels))
    
        VM.append(v_measure_score(y,_labels))
        """
        Inertia is only a sensible measure for spherical clusters. I.e. not for DBSCAN. 
        Similar reasonings apply for most internal measures: most are designed around centroid-based cluster models
        , not arbitrarily shaped clusters.
        """
        
    print("n cluster", all_cluster)
    plt.figure()
    plt.title('file_name ' + file_name)
    plt.plot(eps_range,all_ars)
    plt.plot(eps_range, all_cs)
    plt.plot(eps_range, MIS)
    plt.plot(eps_range, H)
    plt.plot(eps_range, VM)
    plt.xticks(eps_range)
    plt.ylabel('Performance')
    plt.xlabel('Eps')
    plt.legend(['Rand Index Score', 'Completeness Score','Mutual information score' ,'Homogeneity' , 'V-measure'], loc='best')
    plt.show()
    
    dbscan = cluster.DBSCAN(eps=8)
    
    birch = cluster.Birch(n_clusters=k_clusters)
    
    """
     Gaussian Mixture
     attempts to find a mixture of multi-dimensional Gaussian probability distributions that best model any input dataset
    """
    
    gmm = mixture.GaussianMixture(n_components=k_clusters, covariance_type='full')
    
    
    #large datasets require smalls quantiles
    
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.2)
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    
    
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
    
    algorithms = agglomerative_algorithms
    algorithms =  algorithms + (
                    ('spectral', spectral),
                    ('affinity', affinity),
                    ('dbscan', dbscan),
                    ('birch', birch),
                    ('gmm', gmm),
                    ('ms', ms),
                    ('birch', birch),
                    ('TM', two_means)
                            )
    
    
    
    xticklabels = []
    all_ars = []
    all_cs = []
    MIS = []
    H = []
    VM = []
    for name, algorithm in algorithms:    
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
        #sum of squared errors TODO
        
    
    fig,ax = plt.subplots()
    x = range(0,len(algorithms))
    plt.bar(x, all_cs)
    plt.xticks(x, xticklabels)
    plt.title('file_name ' + file_name)
    ax.set_ylabel('Completeness score')
    plt.show()
    
    fig,ax = plt.subplots()
    x = range(0,len(algorithms))
    plt.bar(x, MIS)
    plt.xticks(x, xticklabels)
    plt.title('file_name ' + file_name)
    ax.set_ylabel('Mutual information score')
    plt.show()
       
    
    
    fig,ax = plt.subplots()
    x = range(0,len(algorithms))
    plt.bar(x, H)
    plt.xticks(x, xticklabels)
    plt.title('file_name ' + file_name)
    ax.set_ylabel('Homogeneity score')
    plt.show()
       
        
    
    fig,ax = plt.subplots()
    x = range(0,len(algorithms))
    plt.bar(x, VM)
    plt.xticks(x, xticklabels)
    plt.title('file_name ' + file_name)
    ax.set_ylabel('V-measure score')
    plt.show()
      
     
for file_name in os.listdir( r'data\Colposcopy' ):
  

    data = pd.read_csv(
        os.path.join( '.', 'data', 'Colposcopy', file_name ),
        na_values = 'na'
    )
    
    data_o1 = data
    Q1 = data_o1.quantile(0.25)
    Q3 = data_o1.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    print(">>>>>>>>>>>>",file_name)
    data_out = data_o1[~((data_o1 < (Q1 - 1.5 * IQR)) |(data_o1 > (Q3 + 1.5 * IQR))).any(axis=1)]
    data = data_out
    run()