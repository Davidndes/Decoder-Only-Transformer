#gpt 

import torch
import torch.nn as nn # PyTorch's neural network library
from torch.nn import functional as F # makes the functions in torch.nn.functional available as F
import time 

# hyperparameters

batch_size = 64 # number of text sequences processed together in one training step
block_size = 256 # number of tokens (characters) in each sequence, maximum context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available() 
    else 'cpu' # CUDA lets PyTorch run calculations on a supported NVIDIA GPU. therefore this uses an NVIDIA GPU if available, otherwise Apple's GPU backend MPS, otherwise the CPU
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

# create a mapping from characters to integers (tokenizer) using dictionaries
strtoint = { j:i for i,j in enumerate(sorted_characters)}
inttostr = { i:j for i,j in enumerate(sorted_characters)}

encoder = lambda s: [strtoint[vocab_size] for vocab_size in s] # lambda is a "quick" function that can be defined in one line. it takes a string input as s and then returns a list of integers that correspond to the characters in the string.
decoder = lambda l: ''.join(inttostr[d] for d in l) # changes a list of integers back to a string

data = torch.tensor(encoder(text), dtype = torch.long)

# splitting into training and validation data sets
n = int(0.9*len(data))
training_data = data[:n]
validation_data = data[n:]

torch.manual_seed(1337) # random but repeatable 1337 sets the initial state
def get_batch(split): # defines a function called get_batch that takes a string input called split, telling it which dataset to use 
    data = training_data if split == 'train' else validation_data # so calling get_batch('train') will use training dataset
    ix = torch.randint(len(data)-block_size, (batch_size,)) # this means pick four random integers from 0 through 91 each of which represents the starting of a sequence
    # batch_size,) is a tuple that tells PyTorch to return a tensor of the size batch_size 
    x = torch.stack([data[i:i+block_size] for i in ix]) # this means go to position i in the data tensor and take the next block_size characters and do this for each of the four random integers in ix. 
    # torch.stack takes a list of tensors and combines them into a single tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device) # .to(device) puts the tokens on the device selected (CPU or CUDA GPU)
input_tokens, target_tokens = get_batch('train') # pulls training data according to the above function

# models

class Head(nn.Module): # this will denote one head of self attention 
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # register_buffer creates a tril variable and assigns it to the module, storing the causal mask as non-trainable model state that moves with the model
        self.dropout = nn.Dropout(dropout)
        self.k_cache = None # these caches will stored previously learned key and values during generation 
        self.v_cache = None

    def cache_reset(self):
        self.k_cache = None
        self.v_cache = None 

    # scaled dot product attention
    def forward(self, example_tensor, use_cache=False): # use_cache enables reuse of earlier keys and values during generation 
        xbatch_size, xsequence_length, xfeature_dim = example_tensor.shape
        k = self.key(example_tensor)
        q = self.query(example_tensor)
        v = self.value(example_tensor)

        # if the cache is empty this store the keys and values just calculated 
        # if earlier exist, it joins the new one to them and caculates the combined result (in the sequence dimension given this adds positions)
        if use_cache:
            if self.k_cache is not None:
                k = torch.cat((self.k_cache, k), dim =1)
                v = torch.cat((self.v_cache, v), dim=1)
            self.k_cache = k
            self.v_cache = v 

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # divides every query-key score by the square root of the head size 
        # select the mask rows for the positions currently being processed 
        cache_length = k.shape[1] - xsequence_length # how many positions were cached before the call
        mask = self.tril[cache_length:cache_length + xsequence_length, :k.shape[1]] # selects the rows and columns relevant for updating 
        wei = wei.masked_fill(mask == 0, float('-inf')) # produces True whereever attention should be blocked 
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei) # this lets the model randomly drop some attention connections during training to reduce overfitting 
        weighted_output_tensor = wei @ v
        return weighted_output_tensor

class MultiHeadAttention(nn.Module): # this runs several heads on the same input. each head has its own learned query, key and value weights, so they can learn different ways to combine that info
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # creates separate attention heads and register them so PyTorch tracks their parameters
        self.projection = nn.Linear(n_embed, n_embed) # the purpose of the projection is to let information from different heads contribute to the same output number. 
        self.dropout = nn.Dropout(dropout) # apply dropout to the projected features during training to reduce overfitting 

    def forward(self, example_tensor, use_cache=False):
        projection = torch.cat([head(example_tensor, use_cache=use_cache) for head in self.heads], dim = -1)
        output = self.dropout(self.projection(projection)) # combine information across heads, then apply dropout
        return output

class FeedForward(nn.Module): 
    # Apply the same nonlinear network separately at each position,
    # allowing it to learn relationships among features beyond a single affine transformation.       
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # expansion here gives the network more intermediate features, increasing capacity to learn different transformations
            nn.ReLU(), 
            nn.Linear(4 * n_embed, n_embed), 
            nn.Dropout(dropout))

    def forward(self, example_tensor):
        return self.net(example_tensor)

class Block(nn.Module):
    # transformer block: communication followed by computation
    def __init__(self, n_embed, n_head): # where n_embed is the embedding dimensions, n_head is the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.feedfwd = FeedForward(n_embed)
        self.layernorm1 = nn.LayerNorm(n_embed) # normalize each token's feature vector before attention to make training more stable 
        self.layernorm2 = nn.LayerNorm(n_embed) 
    
    def forward(self, example_tensor, use_cache=False):
        example_tensor = example_tensor + self.sa(self.layernorm1(example_tensor), use_cache=use_cache) # add attention's learned adjustment to the existing representation
        example_tensor = example_tensor + self.feedfwd(self.layernorm2(example_tensor)) # add the feedforward network's adjustment to the updated representation 
        return example_tensor

def synchronize_device():
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

class TransformerLanguageModel(nn.Module): # inherits from nn.Module PyTorch's model machinery like tracking trainable parameters
    def __init__(self, n_embed): # n_embed is the number of features in each token embedding 
        super().__init__() # initialize the nn.Module
        self.token_embedding_table = nn.Embedding (vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # this lets the model learn a vector for each context position so the model can use character order
        self.blocks = nn.Sequential(*[Block(n_embed, n_head = n_head) for _ in range (n_layer)]) # creates six transformer blocks and runs them one after the other so each further processes the previous' outputs 
        self.layernormf = nn.LayerNorm(n_embed) #final layer norm 
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def cache_reset(self):
        for block in self.blocks:
            for head in block.sa.heads:
                head.cache_reset()

    def forward(self, idx, targets=None, use_cache=False, position_offset=0): # idx takes input_tokens and targets takes target_tokens as the correct answers 
        batch_size, sequence_length = idx.shape
        token_embeddings = self.token_embedding_table(idx) # looks up the learned feature vector for each input token 
        positions = torch.arange(position_offset, position_offset + sequence_length, device = idx.device)
        position_embeddings = self.position_embedding_table(positions)
        
        embeddings_sum = token_embeddings + position_embeddings # this lets the model combine character identity and position information at each position  
        
        for block in self.blocks:
            embeddings_sum = block(embeddings_sum, use_cache=use_cache)
        embeddings_sum = self.layernormf(embeddings_sum)# It normalizes each token’s features before the final layer converts them into character scores (since prior normalization happened before attention)
        logits = self.lm_head(embeddings_sum) # turns the token's internal representations into a score for every possible next character

        if targets is None:
            loss = None # so if we don't have correct answers we don't calculate the loss
        else:
            batch_size,sequence_length ,vocab_size = logits.shape #pulls out the shape of the logits tensor 
            logits = logits.view(batch_size*sequence_length , vocab_size) # reshapes the logits tensor into a 2D table with batch_size*sequence_length  rows and vocab_size columns
            # this puts the predictions in the format of one row of scores per prediction
            targets = targets.view(batch_size*sequence_length ) # reshapes the targets tensor into a 1D vector with batch_size*sequence_length  elements
            loss = F.cross_entropy(logits, targets) # averages the negative log-probability of correct next characters across all batch positions  
        return logits, loss

    def generate_uncached(self, token_sequence, max_new_tokens): # this is what will extend the starting sequence by generating tokens
        context_limit = self.position_embedding_table.num_embeddings # num_embeddings lets us read the maximum context length from the position embedding table 
        # self is the model instance
        # token_sequence is the starting context of indices (tokens)
        # max_new_tokens is how many new tokens to generate
        for _ in range (max_new_tokens): # means we are repeating the work a max_new_tokens number of times.
            # the _ means we are ignoring the loop variable since we don't need it
            token_context = token_sequence[:, -context_limit:] # use only the most recent tokens that fit within the model's context window 
            logits, loss = self(token_context) # self is the model instance, this runs the forward pass (without targets we are only generating text without checking against known answers)
            logits = logits[:, -1, :] # keeps the final position's character scores for every sequence in the batch 
            probabilities = F.softmax(logits, dim = -1) # softmax turns the character scores into probabilities for each sequence 
            # dim = -1 tells softmax to use the last dimension of logits which contains the scores for each possible next character
            next_token = torch.multinomial(probabilities, num_samples = 1) # multinomial makes a random draw when each sample has different probabilities of being picked and num_samples tells it how many to draw
            token_sequence = torch.cat((token_sequence, next_token), dim = 1) # concatenates the new token along the dimension of the sequence so that the new token is added to the end of the sequence
        return token_sequence 
    
    def generate(self, token_sequence, max_new_tokens):
        self.cache_reset()
        context_limit = self.position_embedding_table.num_embeddings
        cache_length = 0 # how many tokens already have keys and values stored

        for _ in range (max_new_tokens): 
            if cache_length == 0: # handles the iteration when the cache is empty
                token_context = token_sequence[:, -context_limit:]
                logits, _ = self(token_context, use_cache=True, position_offset=0)
                cache_length = token_context.shape[1]
            elif cache_length < context_limit: # handles later iterations while there's room in the cache
                token_context = token_sequence[:,-1:]
                logits, _ = self(token_context, use_cache=True, position_offset=cache_length)
                cache_length += 1
            else: # handles a full cache 
                self.cache_reset()
                token_context = token_sequence[:, -context_limit:]
                logits, _ = self(token_context, use_cache=True, position_offset=0)
                cache_length = token_context.shape[1]

            next_token_logits = logits[:,-1,:]  
            probabilities = F.softmax(next_token_logits, dim=-1) # softmax turns the character scores into probabilities for each sequence 
            next_token = torch.multinomial(probabilities, num_samples=1) # multinomial makes a random draw when each sample has different probabilities of being picked and num_samples tells it how many to draw
            token_sequence = torch.cat((token_sequence, next_token), dim=1)
        return token_sequence  




m = TransformerLanguageModel(n_embed).to(device) # creates a model instance

#m.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True)) -- load saved weights

logits, loss = m(input_tokens, target_tokens) # passing in the sequences with the correct answers to run the forward pass
# this only runs the forward pass because PyTorch defines calling a model with parentheses as calling the forward method

optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate) # different from standard adam in how it handles weight decay
 

# training  

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
print(logits.shape) # this prints the returned scores dimensions (prediction positions each with 65 character scores)
print(loss) # prints the final training batch's loss

torch.save(m.state_dict(),"model_weights.pth")

# text generation

m.eval() # turns off dropout

with torch.no_grad(): # stops tracking calculations for learning, saving memory while generating 
    context = torch.zeros((1,1), dtype=torch.long, device=device)

    # without caching 
    synchronize_device()
    start = time.perf_counter()
    uncached_output = m.generate_uncached(context, max_new_tokens=2000)

    synchronize_device()
    uncatched_seconds = time.perf_counter() - start

    # with caching 
    synchronize_device()
    start = time.perf_counter()
    cached_output = m.generate(context, max_new_tokens=2000)

    synchronize_device()
    cached_seconds = time.perf_counter() - start

print(f"Without cache: {uncatched_seconds:.3f} seconds")
print(f"With cache: {cached_seconds:.3f} seconds")

print(decoder(cached_output[0].tolist()))
# this starts with one token and then generates 2000 from it and changes it to a python list from a tensor with zero selecting the first sequence which is passed into decoder
