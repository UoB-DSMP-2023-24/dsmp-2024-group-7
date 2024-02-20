import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tcrdist.repertoire import TCRrep
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def readdata():
    # 读取txt文件
    file_path = 'vdjdb.txt'
    df = pd.read_csv(file_path, delimiter='\t')  # 如果文件使用制表符分隔，可以调整delimiter参数

    # 显示行数和列数
    num_rows, num_cols = df.shape
    print(f"行数: {num_rows}")
    print(f"列数: {num_cols}")

    # 显示列的名字
    column_names = df.columns.tolist()
    print("列的名字:", column_names)

    # 将DataFrame保存为CSV文件，假设保存文件名为output.csv
    output_file = 'vdjdb_csv.csv'
    df.to_csv(output_file, index=False)
    print(f"DataFrame已保存为CSV文件: {output_file}")
    return df["v.segm"]

def TCRDist():
    tcr_repertoire_alpha = TCRrep(
        cell_df=df,
        organism='human',  # 或 'mouse'，根据您的数据集
        chains=['alpha'],  # 或 'beta'，根据您的数据集
        db_file='tcrdist/db/mouse_a_beta_db.tsv',  # 或 'human_beta_db.tsv'，根据您的数据集
    )

    # 计算基因片段相似性
    tcr_repertoire_alpha.compute_sparse_rect_distances()

    # 获取基因片段距离矩阵
    distances_matrix = tcr_repertoire_alpha.rw_beta.divergence

    # distances_matrix现在包含了基因片段之间的相似性/距离信息
    print(distances_matrix)

def WE_Encoding(data):
    # 示例数据

    # 创建一个嵌入层
    embedding_dim = 2  # 嵌入向量的维度
    embedding_layer = tf.keras.layers.Embedding(input_dim=len(set(data)), output_dim=embedding_dim)

    # 将分类变量转换为整数编码
    label_encoder = LabelEncoder()
    encoded_data = label_encoder.fit_transform(data)

    # 将整数编码的数据作为嵌入层的输入
    embedded_data = embedding_layer(tf.constant(encoded_data))

    print(embedded_data.numpy())

class Standard_Dataset(Dataset):
    def __init__(self, data, char_to_idx, max_seq_length):
        self.data = data
        self.char_to_idx = char_to_idx
        self.max_seq_length = max_seq_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sequence = self.data[idx]
        sequence_idx = [self.char_to_idx[char] for char in sequence if char in self.char_to_idx]
        # 如果序列长度不够，则进行填充
        sequence_idx += [0] * (self.max_seq_length - len(sequence_idx))
        return torch.tensor(sequence_idx)

def Transformer(data):
    data = data.astype(str)
    chars = set(''.join(data.dropna()))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    max_seq_length = max(len(seq) for seq in data)
    dataset = Standard_Dataset(data, char_to_idx, max_seq_length)
    dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

    class Transformer_Model(nn.Module):
        def __init__(self, input_size, d_model, nhead, numlayers, dim_feedforward, dropout = 0.1):
            super(Transformer_Model, self).__init__()
            self.embedding = nn.Embedding(input_size, d_model)
            self.transformer = nn.Transformer(d_model = d_model, nhead = nhead, num_encoder_layers = numlayers, num_decoder_layers = numlayers, dim_feedforward = dim_feedforward, dropout = dropout)
            self.fc_ou = nn.Linear(d_model, input_size)
        def forward(self, src):
            src = self.embedding(src)
            output = self.transformer(src, src)
            output = self.fc_ou(output)
            return output

    model = Transformer_Model(len(chars), d_model = 512, nhead = 8, numlayers = 6, dim_feedforward = 2048)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output.view(-1, output.shape[-1]), batch.view(-1))
        loss.backward()
        optimizer.step()
        print(f'LOSS: {loss.item(): .4f}')
    print("Finished")

if __name__ == '__main__':
    Vsegm = readdata()
    WE_Encoding(Vsegm)
    Transformer(Vsegm)
    #TCRDist()