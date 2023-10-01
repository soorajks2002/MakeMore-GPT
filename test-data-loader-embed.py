import torch
import random

torch.manual_seed(123321)
g = torch.Generator().manual_seed(2147483647 + 10)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_feature_size, output_feature_size):
        super(NeuralNetwork, self).__init__()
        self.l1 = torch.nn.Linear(input_feature_size, 35)
        self.l2 = torch.nn.LeakyReLU()
        self.l3 = torch.nn.Linear(35, output_feature_size)

    def forward(self, x):
        return self.l3(self.l2(self.l1(x)))


model = NeuralNetwork(14, 27)

model_path = "model-data-loader-embedding.pth"
model.load_state_dict(torch.load(model_path))

decoder = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm',
           14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 0: '<>'}

# n_names = int(input("Enter number of names to be generated : "))
n_names = 5
embedder = torch.randn((27, 7))

for i in range(n_names):
    result = []
    inp = [0, 0]

    while True:
        x = torch.tensor(inp)
        x = embedder[x]
        x = x.view(-1)

        y = model(x)
        y = torch.nn.functional.softmax(y, dim=-1)
        y = torch.multinomial(y, num_samples=1, generator=g).item()

        if y:
            result.append(decoder[y])

            inp.append(y)
            inp = inp[1:]

        else:
            break

    print(f"{i+1}. {''.join(result[2:])}")
