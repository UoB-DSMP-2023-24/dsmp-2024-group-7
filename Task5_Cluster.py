import pandas as pd
import numpy as np
import seaborn as sns
from tcrdist.repertoire import TCRrep
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

def get_matrix(df_alpha,df_beta,species):
#     define a function to get the chain matrix
    tr_a = TCRrep(cell_df = df_alpha, # get the alpha chain matrix
            organism = species,
            chains = ['alpha'],
            db_file = 'alphabeta_gammadelta_db.tsv',
            compute_distances=False)
    tr_a.cpus=2
    tr_a.compute_sparse_rect_distances(radius = 50, chunk_size = 100)
    tr_b = TCRrep(cell_df = df_beta,  # get the beta chain matrix
                organism = species,
                chains = ['beta'],
                db_file = 'alphabeta_gammadelta_db.tsv',
                compute_distances=False)
    tr_b.cpus=2
    tr_b.compute_sparse_rect_distances(radius = 50, chunk_size = 100)
    df_merge = pd.merge(df_alpha, df_beta, on='complex.id') # combine alpha chain and beta chain
    df_merge.drop(['species_y'], axis=1, inplace=True) # drop extra column
    df_merge.drop(['antigen.species_y'], axis=1, inplace=True)
    df_merge.rename(columns={'antigen.species_x': 'antigen.species'}, inplace=True)
    tr = TCRrep(cell_df = df_merge, # get the combined chain matrix
            organism = species,
            chains = ['alpha','beta'],
            db_file = 'alphabeta_gammadelta_db.tsv',
            compute_distances=False)
    tr.cpus=2
    tr.compute_sparse_rect_distances(radius = 50, chunk_size = 100)
    combined_rw_distance = tr.rw_alpha + tr.rw_beta # add up the output of the combined chain result
    return tr_a.rw_alpha,tr_b.rw_beta,combined_rw_distance,df_merge

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
    tr_a.compute_sparse_rect_distances(radius = 50, chunk_size = 100)
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

    '''
    tsne = TSNE(n_components = 2, random_state = 42)
    A_tsne = tsne.fit_transform(reduced_mx)
    #B_tsne = tsne.fit_transform(reduced_mx)

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    sns.scatterplot(x=A_tsne[:, 0], y=A_tsne[:, 1], hue=df_ab['epitope_y'], ax=ax)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Visualization of the dimensional reduction')
    ax.legend(labels=df_ab['antigen.gene_y'].unique()[:30], title='Antigen Gene')
    plt.show()
    '''

    umap = UMAP(n_components = 2, random_state = 42)
    A_umap = umap.fit_transform(reduced_mx)
    # B_umap = umap.fit_transform(reduced_mx)

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize = (15, 8))
    sns.scatterplot(x = A_umap[:, 0], y = A_umap[:, 1], hue = df_ab['epitope_y'], ax = ax)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Visualization of the dimensional reduction')
    ax.legend(labels = df_ab['antigen.gene_y'].unique()[:30], title = 'Antigen Gene')
    plt.show()


    return A_umap

def K_MEANS(mx):
    kmeans = KMeans(n_clusters = 5)
    clusters = kmeans.fit_predict(mx)

    # Visualization
    plt.scatter(mx[:, 0], mx[:, 1], c = clusters, cmap='viridis', s=10)
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('KMeans Clustering')
    plt.colorbar(label = 'Cluster')
    plt.show()

    # Silhouette Score(轮廓系数): Higher Silhouette Score indicates better quality of clustering results(越高越好), Values in the range [-1, 1].
    silhouette_avg = silhouette_score(mx, clusters)
    # Calinski-Harabasz index: Higher Calinski-Harabasz index indicates better quality of clustering results(越高越好)
    calinski_harabasz_avg = calinski_harabasz_score(mx, clusters)
    print(clusters)
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
    plt.scatter(mx[:, 0], mx[:, 1], c=clusters, cmap='viridis', s=10)
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
    distances = pairwise_distances(mx, metric = 'euclidean')
    eps = 0.5  # field radius
    min_samples = 5  # minimum sample size
    dbscan = DBSCAN(eps = eps, min_samples = min_samples, metric = 'precomputed')
    clusters = dbscan.fit_predict(distances)

    # Visualization
    plt.scatter(distances[:, 0], distances[:, 1], c = clusters, cmap = 'viridis', s=10)
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('DBSCAN Clustering')
    plt.colorbar(label = 'Cluster')
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
    plt.scatter(mx[:, 0], mx[:, 1], c=clusters, cmap='viridis', s=10)
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
    df = df.drop_duplicates()
    df = df.dropna()
    df = df[df['complex.id'] != 0]
    df = df[df['vdjdb.score'] != 0]
    df = df[['complex.id', 'gene', 'cdr3', 'v.segm', 'j.segm', 'species', 'antigen.epitope', 'antigen.gene', 'antigen.species']]
    # data classificition
    df_homo = df[~df['species'].isin(['MacacaMulatta', 'MusMusculus'])]
    df_homo_alpha = df_homo[df_homo['gene'] == 'TRA'].rename(
        columns={'cdr3': 'cdr3_a_aa', 'v.segm': 'v_a_gene', 'j.segm': 'j_a_gene', 'antigen.epitope': 'epitope'})
    df_homo_beta = df_homo[df_homo['gene'] == 'TRB'].rename(
        columns={'cdr3': 'cdr3_b_aa', 'v.segm': 'v_b_gene', 'j.segm': 'j_b_gene', 'antigen.epitope': 'epitope'})

    df_mouse = df[~df['species'].isin(['MacacaMulatta', 'HomoSapiens'])]
    df_mouse_alpha = df_mouse[df_mouse['gene'] == 'TRA'].rename(
        columns={'cdr3': 'cdr3_a_aa', 'v.segm': 'v_a_gene', 'j.segm': 'j_a_gene', 'antigen.epitope': 'epitope'})
    df_mouse_beta = df_mouse[df_mouse['gene'] == 'TRB'].rename(
        columns={'cdr3': 'cdr3_b_aa', 'v.segm': 'v_b_gene', 'j.segm': 'j_b_gene', 'antigen.epitope': 'epitope'})

    df_monkey = df[~df['species'].isin(['MusMusculus', 'HomoSapiens'])]
    df_monkey_alpha = df_mouse[df_mouse['gene'] == 'TRA'].rename(
        columns={'cdr3': 'cdr3_a_aa', 'v.segm': 'v_a_gene', 'j.segm': 'j_a_gene', 'antigen.epitope': 'epitope'})
    df_monkey_beta = df_mouse[df_mouse['gene'] == 'TRB'].rename(
        columns={'cdr3': 'cdr3_b_aa', 'v.segm': 'v_b_gene', 'j.segm': 'j_b_gene', 'antigen.epitope': 'epitope'})

    # calculate distance matrix
    homo_alpha_matrix, homo_beta_matrix, homo_combined_matrix, df_homo_combined = get_matrix(df_homo_alpha,
                                                                                             df_homo_beta, 'human')
    mouse_alpha_matrix, mouse_beta_matrix, mouse_combined_matrix, df_mouse_combined = get_matrix(df_mouse_alpha,
                                                                                                 df_mouse_beta, 'mouse')

    print(df_mouse_combined['antigen.species_y'].unique())

    # Combined dimensionality reduction
    # df.iloc[:100]
    data_reduced = Combined_Reduction(mouse_combined_matrix, df_mouse_combined)
    #(data_reduced)

    # Silhouette Score(轮廓系数): Higher Silhouette Score indicates better quality of clustering results(越高越好), Values in the range [-1, 1].
    silhouette_avg = []

    # Calinski-Harabasz index: Higher Calinski-Harabasz index indicates better quality of clustering results(越高越好)
    calinski_harabasz_avg = []

    # K-Means Clustering
    ret1, ret2 = K_MEANS(data_reduced)
    silhouette_avg.append(ret1)
    calinski_harabasz_avg. append(ret2)

    # Spectral clustering
    # ret1, ret2 = Spectral(data_reduced)
    # silhouette_avg.append(ret1)
    # calinski_harabasz_avg.append(ret2)

    # DBSCAN clustering
    # ret1, ret2 = DB_SCAN(data_reduced)
    # silhouette_avg.append(ret1)
    # calinski_harabasz_avg.append(ret2)

    # Agglomerative Hierarchical Clustering凝聚层次聚类
    # ret1, ret2 = AHC(data_reduced)
    # silhouette_avg.append(ret1)
    # calinski_harabasz_avg.append(ret2)

    print("K_Means Clustering:")
    print("Silhouette Score:", silhouette_avg[0])
    print("Calinski-Harabasz Index:", calinski_harabasz_avg[0])
    #print("Spectral Clustering:")
    #print("Silhouette Score:", silhouette_avg[1])
    #print("Calinski-Harabasz Index:", calinski_harabasz_avg[1])
    '''
    print("DBSCAN Clustering:")
    print("Silhouette Score:", silhouette_avg[1])
    print("Calinski-Harabasz Index:", calinski_harabasz_avg[1])
    print("Agglomerative Hierarchical Clustering:")
    print("Silhouette Score:", silhouette_avg[2])
    print("Calinski-Harabasz Index:", calinski_harabasz_avg[2])
    '''
'''
SVD+UMAP:
K_Means Clustering:
Silhouette Score: 0.57533884
Calinski-Harabasz Index: 1227.9641350464085
DBSCAN Clustering:
Silhouette Score: 0.5315071
Calinski-Harabasz Index: 288.94587519178236
all:
K_Means Clustering:
Silhouette Score: 0.3286636
Calinski-Harabasz Index: 20815.35727377217
DBSCAN Clustering:
Silhouette Score: -0.1631865
Calinski-Harabasz Index: 391.1579697407055

SVD+TSNE:
K_Means Clustering:
Silhouette Score: 0.3478193
Calinski-Harabasz Index: 69.621394143135
DBSCAN Clustering:
Silhouette Score: 0.11649117
Calinski-Harabasz Index: 22.948673748642335
all:
K_Means Clustering:
Silhouette Score: 0.34258187
Calinski-Harabasz Index: 25088.480705750433
DBSCAN Clustering:
Silhouette Score: -0.5337857
Calinski-Harabasz Index: 71.7691637787613


'''