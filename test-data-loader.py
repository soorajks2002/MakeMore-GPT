import torch
import random


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_feature_size, output_feature_size):
        super(NeuralNetwork, self).__init__()
        self.l1 = torch.nn.Linear(input_feature_size, 35)
        self.l2 = torch.nn.Linear(35, output_feature_size)
        # self.l3 = torch.nn.Softmax()

    def forward(self, x):
        # return self.l3(self.l2(self.l1(x)))
        return self.l2(self.l1(x))


model = NeuralNetwork(54, 27)

model_path = "model-data-loader.pth"
model.load_state_dict(torch.load(model_path))

decoder = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm',
           14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 0: '<>'}

n_names = int(input("Enter number of names to be generated : "))
# n_names = 4

for i in range(n_names):
    result = []

    inp = [random.randint(0, 26)]
    if inp[0]:
        inp.append(random.randint(1, 26))
        result.append(decoder[inp[0]])
    else:
        inp.append(random.randint(0, 26))

    if inp[1]:
        result.append(decoder[inp[1]])

    while True:
        x = torch.tensor(inp)
        x = torch.nn.functional.one_hot(x, num_classes=27).float()
        x = x.view(-1)

        y = model(x)
        y = y.argmax().item()

        if y:
            result.append(decoder[y])

            inp.append(y)
            inp = inp[1:]

        else:
            break

    print(f"{i+1}. {''.join(result)}")
