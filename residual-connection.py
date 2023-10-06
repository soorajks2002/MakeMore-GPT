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
Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)


class NeuralNetwork (torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.embed = torch.nn.Embedding(num_embeddings=27, embedding_dim=4)
        self.flat = torch.nn.Flatten()

        self.l1 = torch.nn.Linear(20, 50)
        self.a1 = torch.nn.Tanh()
        self.l2 = torch.nn.Linear(50, 20)
        self.a2 = torch.nn.Tanh()
        self.l3 = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = self.flat(self.embed(x))
        out = self.a1(self.l1(x))
        out = self.a2(self.l2(out))

        # residual / skip connection
        out = out+x

        out = self.l3(out)

        return out


model = NeuralNetwork()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10

for e in range(epochs):
    out = model(X[:30])
    loss = criterion(out, Y[:30])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch : {e+1}\t Loss : {loss}")
