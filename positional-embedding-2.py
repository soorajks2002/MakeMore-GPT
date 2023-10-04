import torch

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
        self.i_division_values = 1 / torch.pow(10000, 2 * torch.arange(embedding_size) / embedding_size)

    def forward(self, x):
        batch, n_positions = x.shape

        position_matrix = torch.arange(n_positions, dtype=torch.float).unsqueeze(0)

        positional_embedding = position_matrix * self.i_division_values.unsqueeze(1)

        positional_embedding = torch.sin(positional_embedding)  # Applies sin to even indices
        positional_embedding[:, 1::2] = torch.cos(positional_embedding[:, 1::2])  # Applies cos to odd indices

        # return positional_embedding #return in embedding_size, n_positions formta
        return positional_embedding.permute(1,0)



positional_embedder = PositionalEmbedding(10)
pos_embeddings = positional_embedder(X)

print(pos_embeddings[2], pos_embeddings.shape)