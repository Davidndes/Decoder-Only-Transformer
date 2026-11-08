# Character-Level Text Generation with a Transformer

The goal of this personal project is to understand how a transformer learns to generate text by implementing one in PyTorch. I followed Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial, starting with a bigram model and building up to a transformer trained on Tiny Shakespeare.

As with my handwritten digit classifier, I have approached this README as a way to document what I learned and the questions I had. The base architecture follows the tutorial. I extended it by adding KV caching to reuse previously calculated keys and values during text generation, rather than recalculating them for every new character. To measure its effect, I compared generation time with and without caching using the same model, the same starting context, and 250 new characters.

## Summary facts

1. Dataset: Tiny Shakespeare, split into the first 90% for training and the remaining 10% for validation.
2. Model: character-level, decoder-only transformer built in PyTorch.
3. Architecture: token and position embeddings → six transformer blocks → final layer normalization → character scores.
4. Configuration: 384 embedding dimensions, six attention heads per block, and a maximum context of 256 characters.
5. Training: batches of 64 sequences, 5,000 optimizer updates, cross-entropy loss, and AdamW with a learning rate of 0.0003.
6. Dropout: 0.2 during training; disabled during generation.
7. Extension: KV caching, with separate cached and uncached generation methods and a comparison of their execution times.
8. Evaluation: the script prints training-batch loss every 500 updates. Validation evaluation has not yet been implemented.

## Model outline

In my digit classifier, the model received an image and learned to predict its label. Here, the input is a sequence of characters, and the correct answer at each position is the character immediately after it.

During generation, the model produces scores for the possible next characters, softmax converts those scores into probabilities, and a character is sampled from that distribution. It is appended to the sequence, and the process repeats.

### How the model represents a character

First, each distinct character is assigned an integer. This integer only identifies the character and does not describe its meaning or how similar it is to another character.

The token embedding table then maps each integer to a vector of 384 numbers which are the model's representation of a character.

The embedding table contains learned parameters. Looking up a character retrieves its row from that table. As the representation passes through the transformer blocks, its values also depend on the preceding context.

A learned position embedding is added to each token embedding so the model has information about where the character appears in the input. In the current implementation, these positions run from 0 to 255 within the context window.

### What self-attention is doing

The starting point was averaging the representations of the characters seen so far. This lets each position contain information from earlier positions, but it gives every allowed position the same weight.

Self-attention makes those weights depend on the input. Each head applies three separate learned linear transformations to produce queries, keys and values. A query is compared with the keys to produce attention scores. After masking and softmax, those scores become weights used to combine the value vectors.

### Why the causal mask is necessary

While training, the entire input sequence is available at once. Without a restriction, a position could use the next character even though that is the answer it is supposed to predict.

The causal mask allows each position to attend to itself and earlier positions, but prevents it from attending to later ones. In the code, a lower-triangular matrix identifies the allowed positions. Scores for the future positions are replaced with negative infinity before softmax, which gives those positions zero attention weight.

### Why divide by the square root of the head size?

The attention score is a dot product between a query and a key. A larger head means adding more products together. Under the simplifying assumption that the query and key entries are independent, have mean zero and variance one, a dot product with `head_size` terms has variance `head_size`.

Dividing by the square root of `head_size` keeps the scale of those scores more consistent as the head size changes. Otherwise, large differences between scores can make softmax concentrate almost all its weight on one position and produce very small gradients.

In the code, this is the multiplication by `k.shape[-1] ** -0.5`. Each head has 64 dimensions, so the scores are divided by 8. This scaling happens before masking and softmax; it does not itself turn the scores into probabilities.

## Keeping track of the shapes

I renamed several variables while following the tutorial to better articulate what dimension represented. Here, `B` is the number of sequences in a batch and `T` is the number of positions in each sequence.

These shapes describe a full forward pass without an existing cache:

| Stage | Shape | What it contains |
| --- | --- | --- |
| Input tokens | `(B, T)` | One character ID per position |
| Token embeddings plus position embeddings | `(B, T, 384)` | A representation for each position |
| Queries, keys or values in one head | `(B, T, 64)` | A separate transformation of those representations |
| Query-key scores | `(B, T, T)` | A score for each query position against each key position |
| Output from one head | `(B, T, 64)` | Weighted combinations of the value vectors |
| Six head outputs concatenated | `(B, T, 384)` | All six results joined along the feature dimension |
| Final character scores | `(B, T, vocab_size)` | One score for each possible next character |

For a full training batch, `B = 64` and `T = 256`. To calculate cross-entropy, the final scores are reshaped to `(B * T, vocab_size)` and the targets to `(B * T,)`. Each row then corresponds to one next-character prediction and its correct answer.

### What the rest of the transformer block adds

Multihead attention runs six heads with separate learned weights. Their outputs are concatenated and passed through a linear projection, which allows the results from the different heads to be combined. The heads are not assigned particular jobs in advance.

The feedforward network then processes each position separately, using the same network at every position. It expands the representation from 384 to 1,536 numbers, applies ReLU, and maps it back to 384. Attention allows information to pass between positions; the feedforward network transforms the resulting representation at each position. The nonlinearity lets it learn relationships among features beyond a single affine transformation.

Each block also uses layer normalization and residual additions. Layer normalization operates across the features of each token. A residual addition adds the sublayer's output back to its input, so the sublayer can learn an adjustment to the representation. Dropout is used during training as a form of regularization.

Finally, a linear layer maps each 384-number representation to `vocab_size` scores. This is where the internal representation becomes a prediction over characters.

## My extension: Key-Value caching

The motivating question behind this addition was asking "when generating one new character, how much work am I repeating for characters I have already processed"?

In the original generator, each step sends the entire available context through the model again. This recalculates the keys and values for earlier positions, even when those positions and their context have not changed.

I added a separate key and value cache to each attention head in each transformer block. On the first iteration, the model processes the starting context and stores its keys and values. On subsequent iterations, it processes only the newest token and combines its keys and values with the stored ones.

The generation loop still follows the original order: forward pass → sample a character → append it to the sequence.

1. Each attention head now stores its own `k_cache` and `v_cache`.
2. A `use_cache` argument passes through the model, blocks and attention modules. It defaults to `False`, so training continues without caching.
3. The causal mask selects the rows corresponding to the positions currently being processed. For one new token, all cached positions and the token itself are allowed.
4. A `position_offset` gives the new token its correct position instead of restarting it at zero.
5. Each independent generation starts by clearing the caches.
6. The original generator is retained as `generate_uncached` for comparison.

### Why cache keys and values, but not queries?

When processing a new token, its query is compared against the earlier keys and its own key. The resulting attention weights combine the earlier values and its own value.

The earlier queries were used to calculate the earlier positions' outputs. They are not needed to calculate the new position's output, so I only cache keys and values.

### The context-length limit

The model has learned position embeddings for 256 positions. In the original generator, once the sequence becomes longer than that, it uses the most recent 256 tokens and assigns their positions from zero again.

To preserve that behavior, the cached generator clears and rebuilds its cache when the window shifts. 

## Timing comparison

I compared the two generation methods using the same model weights, a starting context containing token ID zero, and 250 new characters. Both calls used evaluation mode with gradient tracking disabled.

GPU operations were synchronized before starting and stopping each timer. The measurements include processing the starting context and generating the new characters, but exclude decoding and printing.

| Method | Time to generate 250 characters |
| --- | --- |
| Without KV caching | 4.530 seconds |
| With KV caching | 3.337 seconds |

Hardware: MPS

Across three runs generating 250 new characters on Apple’s MPS backend, median throughput increased from 55.2 to 74.9 tokens/sec with KV caching—a 35.8% increase (1.36× speedup). Timing includes prompt processing and excludes decoding and printing. All runs stayed within the 256-token context limit.

## Results and open questions

The current script creates a validation split, but does not yet evaluate loss on it. The loss printed during training comes from the current training batch, so it is not a measure of performance on unseen text.

Further questions include:

1. How consistent is the timing difference across repeated runs?
2. How well does the trained model predict characters in the validation split?
