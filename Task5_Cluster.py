import pandas as pd
import numpy as np
import seaborn as sns
from tcrdist.repertoire import TCRrep
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

def TCR_Dist(dfxa, dfxb):
    '''
     In the tcrdist package, the Random Walk distance is one of the methods used to measure the similarity between TCRs.
     It is calculated based on the number of shortest paths between TCR sequences and reflects the structural and sequence similarity between TCRs.

     In the above code, tr_a is a TCRrep object that disables the automatic calculation of distances by setting the compute_distances=False parameter.
     Then the compute_sparse_rect_distances method is called to manually compute the distances between alpha chains and store the results in the rw_alpha property.
     This distance matrix can be obtained by printing tr_a.rw_alpha at the end.
    '''
    tr_a = TCRrep(cell_df = dfxa,
                  organism = 'human',
                  chains = ['alpha'],
                  db_file = 'alphabeta_gammadelta_db.tsv',
                  compute_distances = False)
    tr_a.cpus = 2
    tr_a.compute_sparse_rect_distances(radius=50, chunk_size=100)
    return tr_a.rw_alpha

def SVD_Reduction(mx):
    '''
      PCA downscaling is only supported for sparse inputs using the "arpack" solver, and I'm passing "auto" here. This may have resulted in an error.
      To work around this, consider using the TruncatedSVD class instead of PCA, as it is specifically designed to work with sparse matrices.
    '''
    svd = TruncatedSVD(n_components = 2)
    data_ret = svd.fit_transform(mx)

    # Visualization
    plt.scatter(data_ret[:, 0], data_ret[:, 1])
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('SVD Dimensionality Reduction')
    plt.show()

    return data_ret

def Combined_Reduction(mx, df_ab):
    '''
     first downscaled to 50 dimensions using SVD and then downscaled to 2 dimensions using SVD to achieve greater efficiency.
    '''
    svd = TruncatedSVD(n_components = 50)
    reduced_mx = svd.fit_transform(mx)
    explained_variance_ratio = svd.explained_variance_ratio_

    pca = PCA(n_components = 2)
    pca_mx = pca.fit_transform(reduced_mx)

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize = (15, 8))
    sns.scatterplot(x = pca_mx[:, 0], y = pca_mx[:, 1], hue = df_ab['epitope'], ax = ax)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Visualization of the dimensional reduction')
    ax.legend(labels = df_ab['antigen.gene'].unique()[:30], title = 'Antigen Gene')
    plt.show()

    return pca_mx

def K_MEANS(mx):
    kmeans = KMeans(n_clusters = 5)
    clusters = kmeans.fit_predict(mx)

    # Visualization
    plt.scatter(mx[:, 0], mx[:, 1], c = clusters, cmap='viridis')
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('KMeans Clustering')
    plt.colorbar(label = 'Cluster')
    plt.show()

    # Silhouette Score(轮廓系数): Higher Silhouette Score indicates better quality of clustering results(越高越好), Values in the range [-1, 1].
    silhouette_avg = silhouette_score(mx, clusters)
    # Calinski-Harabasz index: Higher Calinski-Harabasz index indicates better quality of clustering results(越高越好)
    calinski_harabasz_avg = calinski_harabasz_score(mx, clusters)
    return silhouette_avg, calinski_harabasz_avg

def Spectral(mx):
    # Compute different types of distances
    distances = pairwise_distances(mx, metric = 'euclidean')
    # distances = pairwise_distances(mx, metric = 'manhattan')
    # distances = pairwise_distances(mx, metric = 'chebyshev')
    # distances = pairwise_distances(mx, metric = 'jaccard')
    # distances = pairwise_distances(mx, metric = 'correlation')
    # distances = pairwise_distances(mx, metric = 'seuclidean')

    # Perform spectral clustering
    spectral_clustering = SpectralClustering(n_clusters = 5, affinity='precomputed')
    clusters = spectral_clustering.fit_predict(distances)

    # Visualization
    plt.scatter(mx[:, 0], mx[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('Spectral Clustering')
    plt.colorbar(label = 'Cluster')
    plt.show()

    # Silhouette Score(轮廓系数): Higher Silhouette Score indicates better quality of clustering results(越高越好), Values in the range [-1, 1].
    silhouette_avg = silhouette_score(distances, clusters)

    # Calinski-Harabasz index: Higher Calinski-Harabasz index indicates better quality of clustering results(越高越好)
    calinski_harabasz_avg = calinski_harabasz_score(distances, clusters)
    return silhouette_avg, calinski_harabasz_avg

def DB_SCAN(mx):
    mx[mx < 0] = 0
    distances = pairwise_distances(mx, metric='euclidean')
    eps = 0.5  # field radius
    min_samples = 5  # minimum sample size
    dbscan = DBSCAN(eps = eps, min_samples = min_samples, metric = 'precomputed')
    clusters = dbscan.fit_predict(distances)

    # Visualization
    plt.scatter(distances[:, 0], distances[:, 1], c = clusters, cmap='viridis')
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('DBSCAN Clustering')
    plt.colorbar(label='Cluster')
    plt.show()

    # Silhouette Score(轮廓系数): Higher Silhouette Score indicates better quality of clustering results(越高越好), Values in the range [-1, 1].
    silhouette_avg = silhouette_score(distances, clusters)

    # Calinski-Harabasz index: Higher Calinski-Harabasz index indicates better quality of clustering results(越高越好)
    calinski_harabasz_avg = calinski_harabasz_score(distances, clusters)
    return silhouette_avg, calinski_harabasz_avg

def AHC(mx):
    n_clusters = 5  # number of clusters
    agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    clusters = agglomerative_clustering.fit_predict(mx)

    # Visualization
    plt.scatter(mx[:, 0], mx[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('Agglomerative Hierarchical Clustering')
    plt.colorbar(label='Cluster')
    plt.show()

    # Silhouette Score(轮廓系数): Higher Silhouette Score indicates better quality of clustering results(越高越好), Values in the range [-1, 1].
    silhouette_avg = silhouette_score(mx, clusters)

    # Calinski-Harabasz index: Higher Calinski-Harabasz index indicates better quality of clustering results(越高越好)
    calinski_harabasz_avg = calinski_harabasz_score(mx, clusters)
    return silhouette_avg, calinski_harabasz_avg

if __name__ == '__main__':
    # Read the entire file into a DataFrame
    df = pd.read_csv("../vdjdb.txt", delimiter='\t')  # Assuming the file is tab-delimited, adjust if needed
    df = df[['complex.id', 'gene', 'cdr3', 'v.segm', 'j.segm', 'species', 'mhc.a', 'mhc.b', 'antigen.epitope', 'antigen.gene']]

    df_alpha = df[df['gene'] == 'TRA'].rename(columns={'cdr3': 'cdr3_a_aa', 'v.segm': 'v_a_gene', 'j.segm': 'j_a_gene','antigen.epitope':'epitope'})
    df_beta = df[df['gene'] == 'TRB'].rename(columns={'cdr3': 'cdr3_b_aa', 'v.segm': 'v_b_gene', 'j.segm': 'j_b_gene','antigen.epitope':'epitope'})

    df = df[~df['species'].isin(['MusMusculus', 'MacacaMulatta'])]
    df_alpha = df_alpha[~df_alpha['species'].isin(['MusMusculus', 'MacacaMulatta'])]
    df_beta = df_beta[~df_beta['species'].isin(['MusMusculus', 'MacacaMulatta'])]

    # print(df_alpha['species'].unique())
    # print(df_alpha.info())

    df_alpha = df_alpha.dropna()
    df_beta = df_beta.dropna()
    df = df.dropna()
    df_alpha = df_alpha.drop_duplicates()
    df_beta = df_beta.drop_duplicates()
    df = df.drop_duplicates()
    # print(df_alpha.info())
    # print(df_beta.info())
    # print(df.info())

    # try to test on a certain number of data
    # df_alpha = df_alpha.iloc[:100]
    # df_beta = df_beta.iloc[:100]

    # calculate distance
    distance_matrix = TCR_Dist(df_alpha, df_beta)
    print(distance_matrix)

    # Combined dimensionality reduction
    data_reduced = Combined_Reduction(distance_matrix, df_alpha)
    print(data_reduced)

    # TruncatedSVD dimensionality reduction
    # data_reduced = SVD_Reduction(distance_matrix)

    # Silhouette Score(轮廓系数): Higher Silhouette Score indicates better quality of clustering results(越高越好), Values in the range [-1, 1].
    silhouette_avg = []

    # Calinski-Harabasz index: Higher Calinski-Harabasz index indicates better quality of clustering results(越高越好)
    calinski_harabasz_avg = []

    # K-Means Clustering
    ret1, ret2 = K_MEANS(data_reduced)
    silhouette_avg.append(ret1)
    calinski_harabasz_avg. append(ret2)

    # Spectral clustering
    ret1, ret2 = Spectral(data_reduced)
    silhouette_avg.append(ret1)
    calinski_harabasz_avg.append(ret2)

    # DBSCAN clustering
    ret1, ret2 = DB_SCAN(data_reduced)
    silhouette_avg.append(ret1)
    calinski_harabasz_avg.append(ret2)

    # Agglomerative Hierarchical Clustering凝聚层次聚类
    ret1, ret2 = AHC(data_reduced)
    silhouette_avg.append(ret1)
    calinski_harabasz_avg.append(ret2)

    print("K_Means Clustering:")
    print("Silhouette Score:", silhouette_avg[0])
    print("Calinski-Harabasz Index:", calinski_harabasz_avg[0])
    print("Spectral Clustering:")
    print("Silhouette Score:", silhouette_avg[1])
    print("Calinski-Harabasz Index:", calinski_harabasz_avg[1])
    print("DBSCAN Clustering:")
    print("Silhouette Score:", silhouette_avg[2])
    print("Calinski-Harabasz Index:", calinski_harabasz_avg[2])
    print("Agglomerative Hierarchical Clustering:")
    print("Silhouette Score:", silhouette_avg[3])
    print("Calinski-Harabasz Index:", calinski_harabasz_avg[3])