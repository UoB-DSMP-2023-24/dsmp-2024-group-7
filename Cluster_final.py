import pandas as pd
import numpy as np
import seaborn as sns
from tcrdist.repertoire import TCRrep
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, fowlkes_mallows_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


def get_matrix(df_alpha, df_beta, species):
    #     define a function to get the chain matrix
    tr_a = TCRrep(cell_df=df_alpha,  # get the alpha chain matrix
                  organism=species,
                  chains=['alpha'],
                  db_file='alphabeta_gammadelta_db.tsv',
                  compute_distances=False)
    tr_a.cpus = 2
    tr_a.compute_sparse_rect_distances(radius=50, chunk_size=100)
    tr_b = TCRrep(cell_df=df_beta,  # get the beta chain matrix
                  organism=species,
                  chains=['beta'],
                  db_file='alphabeta_gammadelta_db.tsv',
                  compute_distances=False)
    tr_b.cpus = 2
    tr_b.compute_sparse_rect_distances(radius=50, chunk_size=100)
    df_merge = pd.merge(df_alpha, df_beta, on='complex.id')  # combine alpha chain and beta chain
    df_merge.drop(['species_y'], axis=1, inplace=True)  # drop extra column
    df_merge.drop(['antigen.species_y'], axis=1, inplace=True)
    df_merge.rename(columns={'antigen.species_x': 'antigen.species'}, inplace=True)
    tr = TCRrep(cell_df=df_merge,  # get the combined chain matrix
                organism=species,
                chains=['alpha', 'beta'],
                db_file='alphabeta_gammadelta_db.tsv',
                compute_distances=False)
    tr.cpus = 2
    tr.compute_sparse_rect_distances(radius=50, chunk_size=100)
    combined_rw_distance = tr.rw_alpha + tr.rw_beta  # add up the output of the combined chain result
    return tr_a.rw_alpha, tr_b.rw_beta, combined_rw_distance, df_merge


def TCR_Dist(dfxa, dfxb):
    '''
     In the tcrdist package, the Random Walk distance is one of the methods used to measure the similarity between TCRs.
     It is calculated based on the number of shortest paths between TCR sequences and reflects the structural and sequence similarity between TCRs.

     In the above code, tr_a is a TCRrep object that disables the automatic calculation of distances by setting the compute_distances=False parameter.
     Then the compute_sparse_rect_distances method is called to manually compute the distances between alpha chains and store the results in the rw_alpha property.
     This distance matrix can be obtained by printing tr_a.rw_alpha at the end.
    '''
    tr_a = TCRrep(cell_df=dfxa,
                  organism='human',
                  chains=['alpha'],
                  db_file='alphabeta_gammadelta_db.tsv',
                  compute_distances=False)
    tr_a.cpus = 2
    tr_a.compute_sparse_rect_distances(radius=50, chunk_size=100)
    return tr_a.rw_alpha

def Combined_Reduction(mx, df_ab):
    '''
     first downscaled to 50 dimensions using SVD and then downscaled to 2 dimensions using SVD to achieve greater efficiency.
    '''
    svd = TruncatedSVD(n_components=50)
    reduced_mx = svd.fit_transform(mx)
    explained_variance_ratio = svd.explained_variance_ratio_

    tsne = TSNE(n_components=2, random_state=42)
    A_tsne = tsne.fit_transform(reduced_mx)

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.scatterplot(x=A_tsne[:, 0], y=A_tsne[:, 1], hue=df_ab['epitope_y'], ax=ax)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Visualization of the dimensional reduction')
    ax.legend(labels=df_ab['antigen.gene_y'].unique()[:30], title='Antigen Gene')
    plt.show()

    return A_tsne

def cluster_metrics(true_labels, cluster_labels):
    num_samples = len(true_labels)
    num_clusters = len(np.unique(cluster_labels))
    cluster_metrics = []

    # 计算匹配度和纯度
    for cluster in range(num_clusters):
        mask = (cluster_labels == cluster)
        cluster_samples = true_labels[mask]
        unique, counts = np.unique(cluster_samples, return_counts=True)
        max_count_index = np.argmax(counts)
        max_count = counts[max_count_index]
        purity = max_count / np.sum(counts)
        match_rate = max_count / np.sum(mask)
        cluster_metrics.append((match_rate, purity))

    return cluster_metrics

def K_MEANS(mx, true_labels, epitope_x, epitope_y):
    N_Clusters = len(true_labels.unique())
    kmeans = KMeans(n_clusters = N_Clusters)
    clusters = kmeans.fit_predict(mx)
    centers = kmeans.cluster_centers_
    color_dict = {i: plt.cm.viridis(i / 10) for i in range(11)}

    # Visualization
    plt.subplots(1, 1, figsize=(12, 8))
    scatter = plt.scatter(mx[:, 0], mx[:, 1], c = clusters, cmap = 'viridis', s = 5)
    legend1 = plt.legend(*scatter.legend_elements(), title='Clusters', loc = 'upper right')
    plt.gca().add_artist(legend1)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100)

    extra_labels = [
        plt.Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = color_dict[i], markersize = 10, label = f'Cluster {i}')
        for i in range(N_Clusters)]
    plt.legend(handles=extra_labels, title = 'Cluster Labels', loc = 'lower right')

    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('KMeans Clustering')
    plt.legend()
    plt.show()

    # Calculate connectivity matrix
    Z = linkage(mx, 'average')
    # Draw a classification tree diagram
    plt.figure(figsize=(10, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
    )
    plt.show()

    # Match Rate / Purity
    metrics = cluster_metrics(true_labels, clusters)
    for i, (match_rate, purity) in enumerate(metrics):
        print(f"Cluster {i} - Match Rate: {match_rate}, Purity: {purity}")

    # Silhouette Score(轮廓系数): Higher Silhouette Score indicates better quality of clustering results(越高越好), Values in the range [-1, 1].
    print("Silhouette Score:", silhouette_score(mx, clusters))

    # Calinski-Harabasz index: Higher Calinski-Harabasz index indicates better quality of clustering results(越高越好)
    print("Calinski-Harabasz Index:", calinski_harabasz_score(mx, clusters))

    # Davies-Bouldin index: Lower Davies-Bouldin index indicates better quality of clustering results(越低越好)
    print("Davies-Bouldin:", davies_bouldin_score(mx, clusters))

    # fowlkes_mallows_score: Higher fowlkes_mallows_score indicates better quality of clustering results(越高越好)
    print("fowlkes_mallows_score:", fowlkes_mallows_score(true_labels, clusters))

    # Output the epitope that occurs in each cluster
    cluster_epitope_x = {i: set() for i in range(N_Clusters)}
    cluster_epitope_y = {i: set() for i in range(N_Clusters)}
    for i, eptitope in enumerate(epitope_x):
        cluster_epitope_x[clusters[i]].add(eptitope)
    for i, eptitope in enumerate(epitope_y):
        cluster_epitope_y[clusters[i]].add(eptitope)

    for cluster_id, eptitopes_x in cluster_epitope_x.items():
        print(f"Cluster {cluster_id} alpha_Eptitopes:", eptitopes_x)
    for cluster_id, eptitopes_y in cluster_epitope_y.items():
        print(f"Cluster {cluster_id} beta_Eptitopes:", eptitopes_y)

    # Output the most frequent epitope in each cluster
    cluster_epitope_x_counts = {i: {} for i in range(N_Clusters)}
    for i, eptitope in enumerate(epitope_x):
        if eptitope in cluster_epitope_x_counts[clusters[i]]:
            cluster_epitope_x_counts[clusters[i]][eptitope] += 1
        else:
            cluster_epitope_x_counts[clusters[i]][eptitope] = 1

    for cluster_id, eptitope_counts in cluster_epitope_x_counts.items():
        max_epitope = max(eptitope_counts, key=eptitope_counts.get)
        max_count = eptitope_counts[max_epitope]
        print(f"Cluster {cluster_id} Most Frequent Alpha_Eptitope: {max_epitope} (Count: {max_count})")

    cluster_epitope_y_counts = {i: {} for i in range(N_Clusters)}
    for i, eptitope in enumerate(epitope_y):
        if eptitope in cluster_epitope_y_counts[clusters[i]]:
            cluster_epitope_y_counts[clusters[i]][eptitope] += 1
        else:
            cluster_epitope_y_counts[clusters[i]][eptitope] = 1

    for cluster_id, eptitope_counts in cluster_epitope_y_counts.items():
        max_epitope = max(eptitope_counts, key=eptitope_counts.get)
        max_count = eptitope_counts[max_epitope]
        print(f"Cluster {cluster_id} Most Frequent Beta_Eptitope: {max_epitope} (Count: {max_count})")

def AHC(mx, true_labels, epitope_x, epitope_y):
    N_Clusters = len(true_labels.unique())  # number of clusters
    agglomerative_clustering = AgglomerativeClustering(n_clusters = N_Clusters, linkage='average')
    clusters = agglomerative_clustering.fit_predict(mx)
    color_dict = {i: plt.cm.viridis(i / 10) for i in range(11)}

    # Visualization
    plt.subplots(1, 1, figsize=(12, 8))
    scatter = plt.scatter(mx[:, 0], mx[:, 1], c=clusters, cmap='viridis', s=5)
    legend1 = plt.legend(*scatter.legend_elements(), title='Clusters', loc='upper right')
    plt.gca().add_artist(legend1)

    extra_labels = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[i], markersize=10, label=f'Cluster {i}')
        for i in range(N_Clusters)]
    plt.legend(handles=extra_labels, title='Cluster Labels', loc='lower right')

    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('Agglomerative Hierarchical Clustering')
    plt.legend()
    plt.show()

    # Calculate connectivity matrix
    Z = linkage(mx, 'average')
    # Draw a classification tree diagram
    plt.figure(figsize=(10, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
    )
    plt.show()

    # Match Rate / Purity
    metrics = cluster_metrics(true_labels, clusters)
    for i, (match_rate, purity) in enumerate(metrics):
        print(f"Cluster {i} - Match Rate: {match_rate}, Purity: {purity}")

    # Silhouette Score(轮廓系数): Higher Silhouette Score indicates better quality of clustering results(越高越好), Values in the range [-1, 1].
    print("Silhouette Score:", silhouette_score(mx, clusters))

    # Calinski-Harabasz index: Higher Calinski-Harabasz index indicates better quality of clustering results(越高越好)
    print("Calinski-Harabasz Index:", calinski_harabasz_score(mx, clusters))

    # Davies-Bouldin index: Lower Davies-Bouldin index indicates better quality of clustering results(越低越好)
    print("Davies-Bouldin:", davies_bouldin_score(mx, clusters))

    # fowlkes_mallows_score: Higher fowlkes_mallows_score indicates better quality of clustering results(越高越好)
    print("fowlkes_mallows_score:", fowlkes_mallows_score(true_labels, clusters))

    # Output the epitope that occurs in each cluster
    cluster_epitope_x = {i: set() for i in range(N_Clusters)}
    cluster_epitope_y = {i: set() for i in range(N_Clusters)}
    for i, eptitope in enumerate(epitope_x):
        cluster_epitope_x[clusters[i]].add(eptitope)
    for i, eptitope in enumerate(epitope_y):
        cluster_epitope_y[clusters[i]].add(eptitope)

    for cluster_id, eptitopes_x in cluster_epitope_x.items():
        print(f"Cluster {cluster_id} alpha_Eptitopes:", eptitopes_x)
    for cluster_id, eptitopes_y in cluster_epitope_y.items():
        print(f"Cluster {cluster_id} beta_Eptitopes:", eptitopes_y)

    # Output the most frequent epitope in each cluster
    cluster_epitope_x_counts = {i: {} for i in range(N_Clusters)}
    for i, eptitope in enumerate(epitope_x):
        if eptitope in cluster_epitope_x_counts[clusters[i]]:
            cluster_epitope_x_counts[clusters[i]][eptitope] += 1
        else:
            cluster_epitope_x_counts[clusters[i]][eptitope] = 1

    for cluster_id, eptitope_counts in cluster_epitope_x_counts.items():
        max_epitope = max(eptitope_counts, key=eptitope_counts.get)
        max_count = eptitope_counts[max_epitope]
        print(f"Cluster {cluster_id} Most Frequent Alpha_Eptitope: {max_epitope} (Count: {max_count})")

    cluster_epitope_y_counts = {i: {} for i in range(N_Clusters)}
    for i, eptitope in enumerate(epitope_y):
        if eptitope in cluster_epitope_y_counts[clusters[i]]:
            cluster_epitope_y_counts[clusters[i]][eptitope] += 1
        else:
            cluster_epitope_y_counts[clusters[i]][eptitope] = 1

    for cluster_id, eptitope_counts in cluster_epitope_y_counts.items():
        max_epitope = max(eptitope_counts, key=eptitope_counts.get)
        max_count = eptitope_counts[max_epitope]
        print(f"Cluster {cluster_id} Most Frequent Beta_Eptitope: {max_epitope} (Count: {max_count})")

if __name__ == '__main__':
    # Read the entire file into a DataFrame
    df = pd.read_csv("../vdjdb.txt", delimiter='\t')  # Assuming the file is tab-delimited, adjust if needed
    df = df.drop_duplicates()
    df = df.dropna()
    df = df[df['complex.id'] != 0]
    df = df[df['vdjdb.score'] != 0]
    df = df[['complex.id', 'gene', 'cdr3', 'v.segm', 'j.segm', 'species', 'antigen.epitope', 'antigen.gene',
             'antigen.species']]
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

    # original gene species tag
    print(np.array(df_mouse_combined.columns))
    true_labels = df_mouse_combined['antigen.species']
    epitope_x = df_mouse_combined['epitope_x']
    epitope_y = df_mouse_combined['epitope_y']

    # Combined dimensionality reduction
    data_reduced = Combined_Reduction(mouse_combined_matrix, df_mouse_combined)

    # Silhouette Score(轮廓系数): Higher Silhouette Score indicates better quality of clustering results(越高越好), Values in the range [-1, 1].
    silhouette_avg = []

    # Calinski-Harabasz index: Higher Calinski-Harabasz index indicates better quality of clustering results(越高越好)
    calinski_harabasz_avg = []

    # K-Means Clustering
    K_MEANS(data_reduced, true_labels, epitope_x, epitope_y)

    # Agglomerative Hierarchical Clustering凝聚层次聚类
    AHC(data_reduced, true_labels, epitope_x, epitope_y)

'''
Cluster 0 - Match Rate: 0.9875, Purity: 0.9875
Cluster 1 - Match Rate: 0.4636363636363636, Purity: 0.4636363636363636
Cluster 2 - Match Rate: 0.9340659340659341, Purity: 0.9340659340659341
Cluster 3 - Match Rate: 0.55, Purity: 0.55
Cluster 4 - Match Rate: 0.6304347826086957, Purity: 0.6304347826086957
Cluster 5 - Match Rate: 0.9078947368421053, Purity: 0.9078947368421053
Cluster 6 - Match Rate: 0.853448275862069, Purity: 0.853448275862069
Cluster 7 - Match Rate: 0.9705882352941176, Purity: 0.9705882352941176
Cluster 8 - Match Rate: 1.0, Purity: 1.0
Cluster 9 - Match Rate: 0.8823529411764706, Purity: 0.8823529411764706
Cluster 10 - Match Rate: 0.9322033898305084, Purity: 0.9322033898305084
Silhouette Score: 0.48569563
Calinski-Harabasz Index: 1273.2121114958677
Davies-Bouldin: 0.7038656661881882
fowlkes_mallows_score: 0.29096532246788254

Cluster 0 - Match Rate: 0.5483870967741935, Purity: 0.5483870967741935
Cluster 1 - Match Rate: 0.4235294117647059, Purity: 0.4235294117647059
Cluster 2 - Match Rate: 0.9880952380952381, Purity: 0.9880952380952381
Cluster 3 - Match Rate: 0.9322033898305084, Purity: 0.9322033898305084
Cluster 4 - Match Rate: 0.9313725490196079, Purity: 0.9313725490196079
Cluster 5 - Match Rate: 0.9705882352941176, Purity: 0.9705882352941176
Cluster 6 - Match Rate: 0.9078947368421053, Purity: 0.9078947368421053
Cluster 7 - Match Rate: 0.5, Purity: 0.5
Cluster 8 - Match Rate: 0.9883720930232558, Purity: 0.9883720930232558
Cluster 9 - Match Rate: 1.0, Purity: 1.0
Cluster 10 - Match Rate: 0.8823529411764706, Purity: 0.8823529411764706
Silhouette Score: 0.46087584
Calinski-Harabasz Index: 1103.6600918851234
Davies-Bouldin: 0.6622617777062516
fowlkes_mallows_score: 0.31287094498942236


'''