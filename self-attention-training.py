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
Y = torch.tensor(Y, dtype=torch.float32)


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.embed = torch.nn.Embedding(27, 10)
        self.attention = torch.nn.MultiheadAttention(embed_dim=10, num_heads=1)
        self.flat = torch.nn.Flatten()
        self.linear = torch.nn.Linear(50,1)

    def __call__(self, x):
        out = self.embed(x)
        out,attention = self.attention(out, out, out)
        out = self.linear(self.flat(out))
        return out


model = NeuralNetwork()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

for e in range(epochs) : 
    out = model(X[:30])
    loss = criterion(out,Y[:30])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Epoch : {e+1}\t Loss : {loss}")