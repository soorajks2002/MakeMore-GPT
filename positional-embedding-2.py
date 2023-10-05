import torch
import math

names = open('names.txt', 'r').read().splitlines()

unique_chars = list(set(''.join(names)))

ctoi = {c: i+1 for i, c in enumerate(unique_chars)}
itoc = {i+1: c for i, c in enumerate(unique_chars)}

ctoi['.'] = 0
itoc[0] = '.'


context_length = 5
X = []
Y = []

for name in names:
    x = [0] * context_length
    for char in (name+'.'):
        X.append(x)
        Y.append(ctoi[char])
        x = x[1:] + [ctoi[char]]

X = torch.tensor(X, dtype=torch.int32)
Y = torch.tensor(Y, dtype=torch.int32)


import torch

class PositionalEmbedding(torch.nn.Module):

    def __init__(self, embedding_size):
        super(PositionalEmbedding, self).__init__()
        self.embedding_size = embedding_size
        i_values = [i-1 if i%2 else i for i in range(embedding_size)]
        self.i_division_values = 1 / torch.pow(10000, torch.tensor(i_values) / embedding_size)

    def forward(self, x):
        _, n_positions = x.shape

        position_matrix = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
        
        positional_embedding = position_matrix * self.i_division_values.unsqueeze(0)
        
        X = positional_embedding
        X[:, 1::2] = torch.cos(X[:, 1::2])
        X[:, 0::2] = torch.sin(X[:, 0::2])
                
        return X


positional_embedder = PositionalEmbedding(5)
pos_embeddings = positional_embedder(X)

print(pos_embeddings)