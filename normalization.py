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


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.embed = torch.nn.Embedding(27, 10)
        self.flat = torch.nn.Flatten()
        self.l1 = torch.nn.Linear(50, 25)
        self.normalize1 = torch.nn.LayerNorm(25)
        self.act1 = torch.nn.Tanh()

        self.l2 = torch.nn.Linear(25, 27)
        self.normalize2 = torch.nn.LayerNorm(27)
        self.act2 = torch.nn.Tanh()

    def __call__(self, x):
        out = self.flat(self.embed(x))
        out = self.act1(self.normalize1(self.l1(out)))
        out = self.act2(self.normalize2(self.l2(out)))
        return out


model = NeuralNetwork()

out = model(X)
print(out.shape)
