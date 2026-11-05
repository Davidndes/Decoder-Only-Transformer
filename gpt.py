#gpt 

import torch
import torch.nn as nn # pytorch's neural network library
from torch.nn import functional as F # makes the functions in torch.nn.functional available as F

# hyperparameters

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available() 
    else 'cpu' # whats is cuda?
)
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

# collecting and organizing the data 

from urllib.request import urlretrieve
urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "input.txt")

with open ('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

sorted_characters = sorted(list(set(text)))
vocab_size = len(sorted_characters)

# create a mapping from chracters to integers (tokenizer) using dictionaries
strtoint = { j:i for i,j in enumerate(sorted_characters)}
inttostr = { i:j for i,j in enumerate(sorted_characters)}

encoder = lambda s: [strtoint[vocab_size] for vocab_size in s] # lambda is a "quick" function that can be defined in one line. it takes a string input as s and then returns a list of integers that correspond to the characters in the string.
decoder = lambda l: ''.join(inttostr[d] for d in l) # changes a list of integers back to a string

data = torch.tensor(encoder(text), dtype = torch.long)

# spltting into training and validation data sets
n = int(0.9*len(data))
training_data = data[:n]
validation_data = data[n:]

torch.manual_seed(1337) # random but repeatable 1337 sets the intial state
# batch_size = 4 # one batch has four individual sequences of text that the model processes together 
# block_size = 8 # each input sequence contains 8 characters / tokens
def get_batch(split): # defines a fucntion called get_batch that takes a string input called split, telling it with dataset to use 
    data = training_data if split == 'train' else validation_data # so calling get_batch('train') will use training dataset
    ix = torch.randint(len(data)-block_size, (batch_size,)) # this means pick four random integers from 0 through 91 each of which represents the startign of a sequence
    # batch_size,) is a tuple that tells pytorch to return a tensor of the size batch_size 
    x = torch.stack([data[i:i+block_size] for i in ix]) # this means go to postion i in the data tensor and take the next block_size characters and do this for each of the four random integers in ix. 
    # torch.stack takes a list of tensors and combines them into a single tensor])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device) # .to(device) puts the tokens on the device selected (CPU or CUDA GPU)
input_tokens, target_tokens = get_batch('train') # pulls training data according tpo the above function

# models 

class Head(nn.Module): # this will denote one head of self attention 
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # register_buffer creates a tril variable and assigns it to the module
        self.dropout = nn.Dropout(dropout)

    # scaled dot product attention
    def forward(self, example_tensor):
        xbatch_size, xsequence_length, xfeature_dim = example_tensor.shape
        k = self.key(example_tensor)
        q = self.query(example_tensor)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # new
        wei = wei.masked_fill(self.tril[:xsequence_length,:xsequence_length] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei) # new, a way to prevent neural net from overfitting? regularization technique
        v = self.value(example_tensor)
        weighted_output_tensor = wei @ v
        return weighted_output_tensor

class MultiHeadAttention(nn.Module): #new 
    # multiple heads of self-attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed) # new
        self.dropout = nn.Dropout(dropout) # new

    def forward(self, example_tensor):
        projection = torch.cat([head(example_tensor) for head in self.heads], dim = -1)
        output = self.dropout(self.projection(projection)) # new 
        return output

class FeedForward(nn.Module):#new 
    # linear layer with a non-linearity        
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), 
            nn.ReLU(), 
            nn.Linear(4 * n_embed, n_embed), 
            nn.Dropout(dropout))

    def forward(self, example_tensor):
        return self.net(example_tensor)

class Block(nn.Module):
    # transformer block: communictaio follwoed by computation
    def __init__(self, n_embed, n_head): # where n_embed is the embedding dimensions, n_head is the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.feedfwd = FeedForward(n_embed)
        self.layernorm1 = nn.LayerNorm(n_embed) # new
        self.layernorm2 = nn.LayerNorm(n_embed) # new
    
    def forward(self, example_tensor):
        example_tensor = example_tensor + self.sa(self.layernorm1(example_tensor))
        example_tensor = example_tensor + self.feedfwd(self.layernorm2(example_tensor))
        return example_tensor

class LayerNorm(): # new
    # makes sure each neuron has unit standrad distribution, what does this mean and why is this necessary?
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps
        #self.momentum = momentum
        #self.training = True
        # parameters trianed with backpropagation
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
       # self.running_mean = torch.zeros(dim)
        #self.running_var = torch.ones(dim)
    def __call__(self, x): #calculate the foward pass
        #if self.training:
        xmean = x.mean(1,keepdim = True) #batch mean
        xvar = x.var(1, keepdim = True)# batch variance
        #else:
         #   xmean = self.running_mean 
          #  xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) #normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        #update the buffers
        #if self.training:
         #   with torch.no_grad():
          #      self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
          #      self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out  

    def parameters(self):
        return [self.gamma, self.beta]

class BigramLanguageModel(nn.Module): # inherits from nn.Module pytorch's model mahcinery like tracking trainable parameters
    def __init__(self, n_embed): # vocab_size tells the function how many distinct charcaters exist
        super().__init__() # initialize the nn.Module
        self.token_embedding_table = nn.Embedding (vocab_size, n_embed)
        # the above creates a square table (vocab_size x vocab_size) which is a trainable lookup tbale 
        # this is a type of neural network layer
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # new
        self.blocks = nn.Sequential(*[Block(n_embed, n_head = n_head) for _ in range (n_layer)]) # new 
        self.layernormf = nn.LayerNorm(n_embed) #final layer norm 
        #self.blocks = nn.Sequential(
        #    Block(n_embed, n_head=4),
         #   Block(n_embed, n_head=4),
          #  Block(n_embed, n_head=4),
          #  nn.LayerNorm(n_embed),
         # new
        #self.sa_heads = MultiHeadAttention(4, n_embed//4) # new (4 heads of 8-dimensional self-attention). like group cpnvolution?  
        #self.feedfwd = FeedForward(n_embed)# new # these eventually got removed but wouldstill like to understand them
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None): # idx takes input_tokens and input tokens and targets takes target_tokens as the correct answers 
        batch_size, sequence_length = idx.shape
        token_embeddings = self.token_embedding_table(idx) # looks up a row of scores for every input token in idx
        position_embeddings = self.position_embedding_table(torch.arange(sequence_length, device = device))
        embeddings_sum = token_embeddings + position_embeddings # new 
        embeddings_sum = self.blocks(embeddings_sum)
        embeddings_sum = self.layernormf(embeddings_sum)# It normalizes each token’s features before the final layer converts them into character scores
        # embeddings_sum = self.feedfwd(embeddings_sum)
        logits = self.lm_head(embeddings_sum) # why did we create a linear layer here?
        # these are in the shape (batch_size, sequence_length , vocab_size)
        # batch_size: number of sequences 
        # sequence_length : tokens (positions) per sequences
        # vocab_size: vocab size
        if targets is None:
            loss = None # so if we don have correct answers we dont calculate the loss
        else:
            batch_size,sequence_length ,vocab_size = logits.shape #pulls out the shape of the logits tensor 
            logits = logits.view(batch_size*sequence_length , vocab_size) # rehapes ths logits tensor into a 2D table with batch_size*sequence_length  rows and vocab_size columns
            # this puts the predictions in the format of one row of scores per prediction
            targets = targets.view(batch_size*sequence_length ) # reshapes the targets tensor into a 1D vector with batch_size*sequence_length  elements
            loss = F.cross_entropy(logits, targets) # measure how much probabality the moeel assigns to the correect next towkn at eahc postion 
        return logits, loss

    def generate(self, token_sequence, max_new_tokens): # this is what will extend the startign sequence by generating tokens
        context_limit = self.position_embedding_table.num_embeddings # new
        # self is the model isntance
        # token_sequence is the starting context of indices (tokens)
        # max_new_tokens is how many new tokens to generate
        for _ in range (max_new_tokens): # means we are repeating the work a max_new_tokens number of times.
            # the _ means we are ignoring the loop variable since we dont need it
            token_context = token_sequence[:, -context_limit:]
            logits, loss = self(token_context) # self is the model instance, this runs the forwrad pass (wihout targets we a re only generating tet withoutchekcing against known ansers)
            logits = logits[:, -1, :] # takes the last oof the logits sequence and the : means we are taking everything along this dimension
            probabilities = F.softmax(logits, dim = -1) # softmax turns the sores for sequence_length  into probabilities for each possible next character
            # dim = -1 tells softmax to use the last dimension of logits which conatins the scores for each possible next character
            next_token = torch.multinomial(probabilities, num_samples = 1) # multinomial makes a random draw when each sample has dfferent proablities of being picked and num_samples tells it how many to draw
            token_sequence = torch.cat((token_sequence, next_token), dim = 1) # concatenates the new oekn aong the dimension of the sequence so that the new token is added to the end of the sequence
        return token_sequence 

m = BigramLanguageModel(n_embed).to(device) # creates a model instance
logits, loss = m(input_tokens, target_tokens) # passing in the sequences with the correct answers to run the forward pass
# this only runs the forward pass because pytorch defines calling a model witt parentheses as calling the forward method



optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate) # different from standard adam in how it handles weight decay
# this tells how many input sequences we train together in one batch
 
m.train()

for steps in range(max_iters): # run the loop max_iter times
    # sample a batch of data
    input_tokens, target_tokens = get_batch('train')
    #evaluate the loss
    logits, loss = m(input_tokens, target_tokens) # runs forward pass method
    optimizer.zero_grad(set_to_none = True) # clears leftover gradients 
    loss.backward()
    optimizer.step()

    if (steps + 1) % eval_interval == 0:
        print(f"Step {steps + 1}/ {max_iters}: loss {loss.item():.4f}")

print(loss.item())
print(logits.shape) # this prints the reurned scores dimensions (prediction posions each with 65 charcater scores)
print(loss) # this tells you how correct the model was across the prediction postions.

m.eval() # turns off dropout

with torch.no_grad(): # stops tracking calculations ofr for learning, saving memory while generating 
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decoder(m.generate(context, max_new_tokens=2000)[0].tolist()))
    # this starts witll one token and then geenrates 100 from it and changes it to a python list from a tensor with zero sleecting the first sequence which is passed into decoder
