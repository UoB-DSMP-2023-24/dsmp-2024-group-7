import pandas as pd
import numpy as np
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
     在 tcrdist 包中，Random Walk 距离是一种用于衡量 TCR 之间相似性的方法之一。
     它是基于 TCR 序列之间的最短路径数量计算得到的，反映了 TCR 之间的结构和序列相似性。

     在上述代码中，tr_a 是一个 TCRrep 对象，通过设置 compute_distances=False 参数来禁用自动计算距离。
     然后调用 compute_sparse_rect_distances 方法手动计算阿尔法链之间的距离，并将结果存储在 rw_alpha 属性中。
     最后打印 tr_a.rw_alpha 可以获取这个距离矩阵。
     '''
    dfxa_first_100 = dfxa.iloc[:100]
    dfxb_first_100 = dfxb.iloc[:100]
    # dfx = pd.merge(dfxa, dfxb, on='complex.id', suffixes=('_alpha', '_beta'))
    # cdr3_a = dfx['cdr3_a_aa']
    # cdr3_b = dfx['cdr3_b_aa']
    # dfx_first_100 = dfx.iloc[:10]

    tr_a = TCRrep(cell_df = dfxa_first_100,
                  organism = 'human',
                  chains = ['alpha'],
                  db_file = 'alphabeta_gammadelta_db.tsv',
                  compute_distances = False)
    tr_a.cpus = 2
    tr_a.compute_sparse_rect_distances(radius=50, chunk_size=100)
    return tr_a.rw_alpha

'''
import pandas as pd
from tcrdist.repertoire import TCRrep

# 读取数据文件
chunk_size = 100000  # 定义每个块的大小
chunks = pd.read_csv("vdjdb.txt", delimiter='\t', chunksize=chunk_size)

# 创建TCRrep对象的列表
tr_a_list = []

# 逐块处理数据
for chunk in chunks:
    # 提取所需列
    chunk = chunk[['complex.id', 'gene', 'cdr3', 'v.segm', 'j.segm', 'species', 'mhc.a', 'mhc.b', 'antigen.epitope', 'antigen.gene']]

    # 分别提取阿尔法链和贝塔链数据
    df_alpha = chunk[chunk['gene'] == 'TRA'].rename(columns={'cdr3': 'cdr3_a_aa', 'v.segm': 'v_a_gene', 'j.segm': 'j_a_gene','antigen.epitope':'epitope'})
    df_beta = chunk[chunk['gene'] == 'TRB'].rename(columns={'cdr3': 'cdr3_b_aa', 'v.segm': 'v_b_gene', 'j.segm': 'j_b_gene','antigen.epitope':'epitope'})

    # 合并阿尔法链和贝塔链数据
    dfx = pd.merge(df_alpha, df_beta, on='complex.id', suffixes=('_alpha', '_beta'))

    # 创建TCRrep对象
    tr_a = TCRrep(cell_df=dfx,
                  organism='human',
                  chains=['alpha', 'beta'],
                  db_file='alphabeta_gammadelta_db.tsv',
                  compute_distances=False)
    
    # 将创建的TCRrep对象添加到列表中
    tr_a_list.append(tr_a)

# 合并所有的TCRrep对象
tr_a_combined = TCRrep.merge(*tr_a_list)
'''

def SVD_Reduction(mx):
    '''
        # PCA降维只支持使用 "arpack" 求解器对稀疏输入进行处理，而我在这里传递了 "auto"。这可能导致了错误。
        # 为了解决这个问题，可以考虑使用 TruncatedSVD 类来代替 PCA，因为它专门用于处理稀疏矩阵。
    '''
    svd = TruncatedSVD(n_components=2)
    data_ret = svd.fit_transform(mx)

    # Visualization
    plt.scatter(data_ret[:, 0], data_ret[:, 1])
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('SVD Dimensionality Reduction')
    plt.show()
    return data_ret

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
    eps = 0.5  # 领域半径
    min_samples = 5  # 最小样本数
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
    n_clusters = 5  # 聚类数量
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

    # calculate distance
    distance_matrix = TCR_Dist(df_alpha, df_beta)
    print(distance_matrix)

    # TruncatedSVD dimensionality reduction
    data_reduced = SVD_Reduction(distance_matrix)

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