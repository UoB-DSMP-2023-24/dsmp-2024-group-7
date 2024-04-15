import pandas as pd
import numpy as np
from tcrdist.repertoire import TCRrep

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
    print("!!!!!!!")
    true_labels = np.array(df_mouse_combined['antigen.species'].unique())
    print(true_labels)
    print(df_mouse_combined.columns)