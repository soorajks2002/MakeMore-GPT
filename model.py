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

for name in text_data:
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
y_data = torch.nn.functional.one_hot(y_data, n_classes).float()

print("x size : ", x_data.shape)
print("y size : ", y_data.shape)


class NeuralNetwork (torch.nn.Module):
    def __init__(self, input_feature_size):
        super(NeuralNetwork, self).__init__()
        self.l1 = torch.nn.Linear(input_feature_size, 50)
        self.l2 = torch.nn.Linear(50, 27)
        self.l3 = torch.nn.Softmax()

    def forward(self, x):
        return self.l2(self.l1(x))
        return self.l3(self.l2(self.l1(x)))


model = NeuralNetwork(x_data.shape[1])
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# print(model(x_data).shape)

epochs = 10

for epoch in range(epochs):
    outputs = model(x_data)
    loss = criterion(outputs, y_data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch : {epoch+1}/{epochs} \t Loss : {loss.data}")


def format_input(vector):
    start_vector = torch.tensor(vector)
    start_vector = embedding_map[start_vector]
    start_vector = start_vector.view(1, -1)
    return start_vector


with torch.no_grad():
    x_input = [0, 0, 0]
    start_vector = format_input(x_input)
    output = []

    while True:
        y_output = model(start_vector)
        y_output = y_output.argmax().item()

        if y_output == 0:
            break

        output.append(itos[y_output])

        x_input.append(y_output)
        x_input = x_input[1:]
        start_vector = format_input(x_input)

    print(''.join(output))
