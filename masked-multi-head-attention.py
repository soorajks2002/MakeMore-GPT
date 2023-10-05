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

        self.embed = torch.nn.Embedding(27, 16)
        self.attention = torch.nn.MultiheadAttention(embed_dim=16, num_heads=8)

    def __call__(self, x):
        out = self.embed(x).permute(1,0,2)
        
        # print(out.shape)
        
        mask = torch.tril(torch.ones(out.size(0), out.size(0)), diagonal=0)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        # print(mask.shape)
        # print(mask)
        
        out,attent = self.attention(out, out, out, attn_mask=mask)
        
        out = out.permute(1,0,2)
        return out, attent


model = NeuralNetwork()

out,attention = model(X[:1])

print(out.shape)
print(attention.shape)