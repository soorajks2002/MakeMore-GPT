import torch

names = open('names.txt', 'r').read().splitlines()
text_data = []

unique_chars = list(set(''.join(names)))

stoi = {char: i+1 for i, char in enumerate(unique_chars)}
itos = {i+1: char for i, char in enumerate(unique_chars)}

stoi['<>'] = 0
itos[0] = '<>'

context_window = 3

for name in names:
    name = ['<>']*context_window + list(name) + ['<>']
    text_data.append(name)

x_data = []
y_data = []

for name in text_data[:10]:
    for x1, x2, x3, y in zip(name[:], name[1:], name[2:], name[3:]):

        x1 = stoi[x1]
        x2 = stoi[x2]
        x3 = stoi[x3]
        y = stoi[y]

        x = [x1, x2, x3]
        x_data.append(x)
        y_data.append(y)

x_data = torch.tensor(x_data)
y_data = torch.tensor(y_data)

embedding_size = 4
embedding_map = torch.randn(size=(27, embedding_size))

x_data = embedding_map[x_data]
x_data = x_data.view(x_data.shape[0], -1)

n_classes = len(stoi)
y_data = torch.nn.functional.one_hot(y_data, n_classes)

print("x size : ", x_data.shape)
print("y size : ", y_data.shape)