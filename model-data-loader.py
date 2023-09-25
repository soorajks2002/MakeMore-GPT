import torch


def get_encoder_decoder(names):
    unique_chars = sorted(list(set(''.join(names))))

    stoi = {char: i+1 for i, char in enumerate(unique_chars)}
    itos = {i+1: char for i, char in enumerate(unique_chars)}

    stoi['<>'] = 0
    itos[0] = '<>'

    return stoi, itos


def encode(data, encoder, context_size):
    encoded_data = []
    for d in data:
        d = ['<>']*context_size + list(d) + ['<>']
        d = [encoder[char] for char in d]
        encoded_data.append(d)
    return encoded_data


def getXY(encoded_data, n_class):
    x = []
    y = []
    for data in encoded_data:
        for x1, x2, y1 in zip(data[0:], data[1:], data[2:]):
            x.append([x1, x2])
            y.append(y1)

    x = torch.tensor(x)
    y = torch.tensor(y)
    x = torch.nn.functional.one_hot(x, n_class).float()
    y = torch.nn.functional.one_hot(y, n_class).float()

    x = x.view(x.shape[0], -1)

    return x, y


class Data (torch.utils.data.Dataset):
    def __init__(self, context_size=2):
        names = open('names.txt', 'r').read().splitlines()
        self.stoi, self.itos = get_encoder_decoder(names)
        encoded_data = encode(names, self.stoi, context_size)
        self.unique_chars = len(self.stoi)
        self.x, self.y = getXY(encoded_data, self.unique_chars)

        self.x_shape = self.x.shape[1]
        self.y_shape = self.y.shape[1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_feature_size, output_feature_size):
        super(NeuralNetwork, self).__init__()
        self.l1 = torch.nn.Linear(input_feature_size, 35)
        self.l2 = torch.nn.Linear(35, output_feature_size)

    def forward(self, x):
        return self.l2(self.l1(x))


data = Data()

model = NeuralNetwork(data.x_shape, data.y_shape)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 10
batch_size = 42000
batchs = int(len(data)/batch_size)

dataloader = torch.utils.data.DataLoader(
    dataset=data, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    print(f"Epoch : {epoch+1}/{epochs}")
    for batch_n, batch_data in enumerate(dataloader):
        x = batch_data[0]
        y = batch_data[1]

        yp = model(x)
        loss = criterion(yp, y)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"\t Batch : {batch_n+1}/{batchs}\t Loss : {loss.data:.2f}")
