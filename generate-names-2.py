import torch

# FIXED - Hyperparameters from TRAINING
context_length = 7
tensor_embedding_size = 30
n_attention_heads = 10  # number of tensor_embedding_size should be divisible by n_attention_heads

learning_rate = 1e-3
batch_size = 22
epochs = 10

# create the integer to character decoder 
names = open('names.txt', 'r').read().splitlines()
unique_chars = list(set(''.join(names)))
itoc = {i+1:c for i,c in enumerate(unique_chars)}
itoc[0] = '.'

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
    def __init__(self, n_unique_chars, embedding_size, context_length, n_heads) :
        super(DecoderNN, self).__init__()

        size = context_length*embedding_size
        self.context_length = context_length

        self.embedder = torch.nn.Embedding(n_unique_chars, embedding_size)
        self.position_encodings = PositionalEncoding(embedding_size)(context_length)
        self.attention = torch.nn.MultiheadAttention(embedding_size, num_heads=n_heads)

        self.flat = torch.nn.Flatten()

        self.linear_1 = torch.nn.Linear(size, size, bias=False)
        self.normalizer_1 = torch.nn.LayerNorm(size)
        self.activation_1 = torch.nn.Tanh()

        self.linear_2 = torch.nn.Linear(size, 300, bias=False)
        self.normalizer_2 = torch.nn.LayerNorm(300)
        self.activation_2 = torch.nn.Tanh()

        self.linear_3 = torch.nn.Linear(300, 75, bias=False)
        self.normalizer_3 = torch.nn.LayerNorm(75)
        self.activation_3 = torch.nn.Tanh()
        
        self.output = torch.nn.Linear(75, n_unique_chars)

    def __call__(self, x) :

        x = self.embedder(x).permute(1,0,2)

        mask = torch.tril(torch.ones(self.context_length, self.context_length))
        mask = mask.masked_fill(mask == 0, float('-inf'))

        x,_ = self.attention(x, x, x, attn_mask=mask)
        x = x.permute(1,0,2)

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
        
        out = self.linear_3(out)
        out = self.normalizer_3(out)
        out = self.activation_3(out)

        out = self.output(out)

        return out
    
# Calculate total trainable parameters in the model
def number_of_parameters(model) :
    num_weights = 0
    for param in model.parameters():
        num_weights += param.numel()
    return num_weights

n_unique_chars = len(itoc)

decoder = DecoderNN(n_unique_chars, tensor_embedding_size, context_length, n_attention_heads)

checkpoint = torch.load('ATTENTION-2.ckpt', map_location=torch.device('cpu'))
decoder.load_state_dict(checkpoint['weights'])


n_names_generate = 20


decoder.eval()  # sets normalization and dropout layers to eval mode
with torch.no_grad() : # stops gradient tracking 
    for i in range(n_names_generate): 
        context = [0]*context_length
        name = []
        while True :
            output = decoder(torch.tensor([context]))
            output = torch.nn.functional.softmax(output, dim=-1)
            gen_char = torch.multinomial(output, num_samples=1)[0][0].item()
            if gen_char :
                name.append(itoc[gen_char])
                context = context[1:] + [gen_char]
            else :
                break
        print(f"{i+1}. {''.join(name)}")