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


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_size):
        super(PositionalEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.i_division_values = [1/(1e4**((2*i)/embedding_size)) for i in range(embedding_size)]
        self.i_division_values = torch.tensor(self.i_division_values)

    def forward(self, x):
        batch, n_positions = x.shape
        
        position_matrix = torch.arange(n_positions)
        position_matrix = position_matrix.unsqueeze(1)
        position_matrix = position_matrix.expand(n_positions, self.embedding_size)
        
        positional_embedding = position_matrix * self.i_division_values
        positional_embedding = [ [ torch.sin(value) if value%2==0 else torch.cos(value) for col,value in enumerate(row)] for row in positional_embedding]
        positional_embedding = torch.tensor(positional_embedding)
        
        return positional_embedding


positional_embedder = PositionalEmbedding(10)
pos_embeddings = positional_embedder(X)

print(pos_embeddings[2], pos_embeddings.shape)