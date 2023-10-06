# TODO :
# ? 1. Create database and dataloader
# ? 2. Create Positional Encoder
# ? 3. Create MultiHead Attention Class
# ? 4. Create Transformer's Decoder Architecture based Model


import torch

# Hyperparameters
context_length = 7
tensor_embedding_size = 15
learning_rate = 1e-3
batch_size = 100
epochs = 100

# load and format the datase
class dataset (torch.utils.data.Dataset):
    def __init__(self, context_length):
        names = open('names.txt', 'r').read().splitlines()
        unique_chars = list(set(''.join(names)))
        self.ctoi = {c:i+1 for i,c in enumerate(unique_chars)}
        self.itoc = {i+1:c for i,c in enumerate(unique_chars)}
        self.ctoi['.'] = 0
        self.itoc[0]   = '.'
        
        self.x = []
        self.y = []
        
        for name in names : 
            chars = [0]*context_length
            for char in name+'.':
                self.x.append(chars)
                self.y.append(self.ctoi[char])
                
                chars = chars[1:] + [self.ctoi[char]]
        
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)
                

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

# Positional Encoding
class PositionalEncoding(torch.nn.Module) :
    def __init__(self, embedding_size) :
        super(PositionalEncoding, self).__init__()
        self.division_term = [i-1 if i%2 else i for i in range(embedding_size)]
        self.division_term = 1/ torch.pow(1e4, torch.tensor(self.division_term)/embedding_size)
        self.division_term = self.division_term.view(1,embedding_size)
        
    def __call__(self, context_length) :
        position_matrix = torch.arange(1, context_length+1)
        position_matrix = position_matrix.view(context_length,1)
        
        position_encoding = position_matrix * self.division_term
        
        position_encoding[:, 0::2] = torch.sin(position_encoding[:,0::2])
        position_encoding[:, 1::2] = torch.cos(position_encoding[:,1::2])
        
        return position_encoding

# Decoder Layer/Architecture from Transformers
class DecoderNN(torch.nn.Module) :
    def __init__(self, n_unique_chars, embedding_size, context_length) :
        super(DecoderNN, self).__init__()
        
        size = context_length*embedding_size
        
        self.embedder = torch.nn.Embedding(n_unique_chars, embedding_size)
        self.position_encodings = PositionalEncoding(embedding_size)(context_length)
        
        self.flat = torch.nn.Flatten()
        
        self.linear_1 = torch.nn.Linear(size, size, bias=False)
        self.normalizer_1 = torch.nn.LayerNorm(size)
        self.activation_1 = torch.nn.Tanh()
        
        self.linear_2 = torch.nn.Linear(size, 25, bias=False)
        self.normalizer_2 = torch.nn.LayerNorm(25)
        self.activation_2 = torch.nn.Tanh()
        
        self.output = torch.nn.Linear(25, n_unique_chars)
        
    def __call__(self, x) :
        
        x = self.embedder(x)
        out = x + self.position_encodings
        
        out = self.flat(out)
        
        out = self.linear_1(out)
        out = self.normalizer_1(out)
        out = self.activation_1(out)
        
        # residual connection
        out = out + self.flat(x)
        
        out = self.linear_2(out)
        out = self.normalizer_2(out)
        out = self.activation_2(out)
        
        out = self.output(out)
        
        return out
    
# Calculate total trainable parameters in the model
def number_of_parameters(model) :
    num_weights = 0
    for param in model.parameters():
        num_weights += param.numel()
    return num_weights

data = dataset(context_length)
batch_loader = torch.utils.data.DataLoader(data, batch_size, shuffle=True)

n_batches = len(batch_loader)
n_unique_chars = len(data.ctoi)

decoder = DecoderNN(n_unique_chars, tensor_embedding_size, context_length)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

for epoch in range(1,epochs+1) :
    print(f"Epoch : {epoch}/{epochs}", end="")
    mean_loss = 0
    for x,y in batch_loader :
        output = decoder(x)
        loss = criterion(output, y)
        mean_loss += loss
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
    print(f"\t Loss : {mean_loss/n_batches}")
    
    if epoch%10 == 0 :
        checkpoint = {'epoch'     : epoch,
                      'optimizer' : optimizer.state_dict(),
                      'weights'   : decoder.state_dict()
                    }
        torch.save(checkpoint, 'NO-ATTENTION.ckpt')