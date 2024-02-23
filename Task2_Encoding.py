import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def readdata():
    # Read txt file
    file_path = 'vdjdb.txt'
    df = pd.read_csv(file_path, delimiter='\t')  # 如果文件使用制表符分隔，可以调整delimiter参数

    # Show number of rows and columns
    num_rows, num_cols = df.shape
    print(f"rows: {num_rows}")
    print(f"columns: {num_cols}")

    # Show the names of the columns
    column_names = df.columns.tolist()
    print("the names of the columns:", column_names)

    # Save the DataFrame as a CSV file and name the file output.csv so that we can observe the data
    output_file = 'vdjdb_csv.csv'
    df.to_csv(output_file, index=False)
    print(f"DataFrame saved as CSV file: {output_file}")
    return df["v.segm"]

#One-hot encoding： Encoding of categorical variables can be readily implemented using Pandas' get_dummies function.
def OH_Encoding(data):
    data = data.astype(str)
    one_hot_encoded = pd.get_dummies(data)
    print(one_hot_encoded)

#Word embedding model: Represents words as low-dimensional vectors with semantic information
def WE_Encoding(data):
    # Create an embedding layer
    embedding_dim = 2
    embedding_layer = tf.keras.layers.Embedding(input_dim=len(set(data)), output_dim=embedding_dim)

    # Convert categorical variables to integer encoding
    label_encoder = LabelEncoder()
    encoded_data = label_encoder.fit_transform(data)

    # Taking integer-encoded data as input to the embedding layer
    embedded_data = embedding_layer(tf.constant(encoded_data))
    print(embedded_data.numpy())

#Tokenizer: Transform unprocessed text into an integer or other numeric representation that the model can understand.
def TK_Encoding(data):
    # Initialize the tokenizer, which refers to a pre-trained model based on the BERT model.
    data = data.astype(str)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Encode string using tokenizer
    encoded_inputs = tokenizer(list(data), padding=True, truncation=True, return_tensors="pt")
    print(encoded_inputs)

#Transformer: The input sequence first goes through the embedding layer, then it goes through the Transformer encoder to extract features, and lastly the fully connected layer outputs the prediction result.
#In parallel, back propagation optimizes parameters while forward propagation computes the loss.
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
    while True:
        x = input("input a number for encoding method! 1 for one-hot, 2 for Word Embedding, 3 for Tokenizer, 4 for Transformer, 5 for quit: ")
        x = int(x)
        if x == 1:
            OH_Encoding(Vsegm)
            print("Finished")
        elif x == 2:
            WE_Encoding(Vsegm)
            print("Finished")
        elif x == 3:
            TK_Encoding(Vsegm)
            print("Finished")
        elif x == 4:
            Transformer(Vsegm)
            print("Finished")
        elif x == 5:
            break
        else:
            print("Invalid input!")