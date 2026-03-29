from collections import defaultdict
import jax
import optax
import pickle
import argparse

def load_data(input_path):
    with open(input_path, "r") as f:
        text = f.read()

    sorted_chars = sorted(set(text))
    char_to_encoding = {c: i for i, c in enumerate(sorted_chars)}
    encoding_to_char = {i: c for i, c in enumerate(sorted_chars)}

    def encode(text: str) -> list[int]:
        return [char_to_encoding[c] for c in text]

    def decode(encoding: list[int]) -> str:
        return "".join([encoding_to_char[i] for i in encoding])

    return text, encode, decode, char_to_encoding, encoding_to_char

CONTEXT_LENGTH = 128
BATCH_SIZE = 32
VOCABULARY = 65

rand_key = jax.random.key(42)

def get_batch(encoded_data_numpy, rng_key):
    starting_positions = jax.random.randint(rng_key, (BATCH_SIZE,), 0, len(encoded_data_numpy) - CONTEXT_LENGTH - 1)

    inputs = []
    outputs = []

    for sp in starting_positions:
        input_data = encoded_data_numpy[sp:sp + CONTEXT_LENGTH + 1]
        output_data = input_data[1:]

        inputs.append(input_data[:-1])
        outputs.append(output_data)

    return (jax.numpy.stack(inputs), jax.numpy.stack(outputs))

rand_key, subkey = jax.random.split(rand_key)

EMBED_DIM = 128

def embed(params, inputs):
    return params['token_embedding'][inputs] + params['positional_embedding']

def attention(params, inputs):
    Q = inputs @ params['W_q']
    K = inputs @ params['W_k']
    V = inputs @ params['W_v']

    # attention scores

    attention_score = Q @ K.transpose(0, 2 ,1)

    # scale attention scores otherwise gradients will vanish

    attention_score_scaled = attention_score / (EMBED_DIM ** 0.5)

    # causal mask so that we only look at the previos tokens

    causal_mask = jax.numpy.triu(jax.numpy.full((CONTEXT_LENGTH, CONTEXT_LENGTH), -jax.numpy.inf), k=1)

    attention_score_scaled_masked = attention_score_scaled + causal_mask

    attention_weights = jax.nn.softmax(attention_score_scaled_masked)

    weighted_sum = attention_weights @ V

    return weighted_sum

NUM_HEADS=4
HEAD_DIM = EMBED_DIM // NUM_HEADS

def multihead_attention(params, inputs):

    batch_size = inputs.shape[0]
    context_length = inputs.shape[1]

    Q = inputs @ params['W_q']
    K = inputs @ params['W_k']
    V = inputs @ params['W_v']

    # reshape from (BATCH, CONTEXT_LENGTH, EMBED_DIM) -> (BATCH, NUM_HEADS, CONTEXT_LENGTH, HEAD_DIMS)

    Q = jax.numpy.reshape(Q, (batch_size, context_length, NUM_HEADS, HEAD_DIM))
    Q = Q.transpose(0, 2, 1, 3)
    K = jax.numpy.reshape(K, (batch_size, context_length, NUM_HEADS, HEAD_DIM))
    K = K.transpose(0, 2, 1, 3)
    V = jax.numpy.reshape(V, (batch_size, context_length, NUM_HEADS, HEAD_DIM))
    V = V.transpose(0, 2, 1, 3)

    # attention scores

    attention_score = Q @ K.transpose(0, 1 ,3, 2)

    # scale attention scores otherwise gradients will vanish

    attention_score_scaled = attention_score / (HEAD_DIM ** 0.5)

    # causal mask so that we only look at the previos tokens

    causal_mask = jax.numpy.triu(jax.numpy.full((context_length, context_length), -jax.numpy.inf), k=1)

    attention_score_scaled_masked = attention_score_scaled + causal_mask

    attention_weights = jax.nn.softmax(attention_score_scaled_masked)

    weighted_sum = attention_weights @ V

    #print(f"{weighted_sum.shape}") (BATCH_SIZE, NUM_HEADS, CONTEXT_LENGTH, HEAD_DIMS)

    # convert back to original shape

    weighted_sum = weighted_sum.transpose(0, 2, 1, 3) # (BATCH_SIZE, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIMS)

    weighted_sum = jax.numpy.reshape(weighted_sum, (batch_size, context_length, EMBED_DIM))

    return weighted_sum @ params['W_o']


def ffn(params, x):
    return jax.nn.relu(x @ params['W1'] + params['b1']) @ params['W2'] + params['b2']

NUM_ATTENTION_BLOCKS = 4    

def init_params(rand_key):
    params = {}
    
    rand_key, subkey = jax.random.split(rand_key)
    # mapping of each token to what its vector representation is
    embedding = jax.random.normal(subkey, (VOCABULARY, EMBED_DIM)) * 0.02

    params['token_embedding'] = embedding

    rand_key, subkey = jax.random.split(rand_key)
    # mapping of context_length to what something at a postion looks like
    positional_embedding = jax.random.normal(subkey, (CONTEXT_LENGTH, EMBED_DIM)) * 0.02

    params['positional_embedding'] = positional_embedding

    params['blocks'] = []

    for i in range(NUM_ATTENTION_BLOCKS):
        block_params = {}

        rand_key, subkey = jax.random.split(rand_key)
        W_q = jax.random.normal(subkey, (EMBED_DIM, EMBED_DIM)) * 0.02

        rand_key, subkey = jax.random.split(rand_key)
        W_k = jax.random.normal(subkey, (EMBED_DIM, EMBED_DIM)) * 0.02

        rand_key, subkey = jax.random.split(rand_key)
        W_v = jax.random.normal(subkey, (EMBED_DIM, EMBED_DIM)) * 0.02

        rand_key, subkey = jax.random.split(rand_key)
        W_o = jax.random.normal(subkey, (EMBED_DIM, EMBED_DIM)) * 0.02

        block_params['W_q'] = W_q
        block_params['W_k'] = W_k
        block_params['W_v'] = W_v
        block_params['W_o'] = W_o

        rand_key, subkey = jax.random.split(rand_key)
        W1 = jax.random.normal(subkey, (EMBED_DIM, 4*EMBED_DIM)) * 0.02
        b1 = jax.numpy.zeros((4*EMBED_DIM,))

        rand_key, subkey = jax.random.split(rand_key)
        W2 = jax.random.normal(subkey, (4*EMBED_DIM, EMBED_DIM)) * 0.02
        b2 = jax.numpy.zeros((EMBED_DIM,))

        block_params["W1"] = W1
        block_params["b1"] = b1
        block_params["W2"] = W2
        block_params["b2"] = b2

        block_params['ln1'] = {
            'gamma': jax.numpy.ones((EMBED_DIM,)),
            'beta': jax.numpy.zeros((EMBED_DIM,)),
        }

        block_params['ln2'] = {
            'gamma': jax.numpy.ones((EMBED_DIM,)),
            'beta': jax.numpy.zeros((EMBED_DIM,)),
        }

        params['blocks'].append(block_params)


    # some nonesense with jit
    params['blocks'] = tuple(params['blocks'])
    # The final W_o needs to take use from embeddings and map to vocabulary (so that model returns the actual output)
    rand_key, subkey = jax.random.split(rand_key)
    W_o = jax.random.normal(subkey, (EMBED_DIM, VOCABULARY)) * 0.02

    params['W_o'] = W_o

    return params

def layer_norm(params, x, eps=1e-5):
    mean = jax.numpy.mean(x, axis=-1, keepdims=True)

    variance = jax.numpy.var(x, axis=-1, keepdims=True)

    x_nomalized = (x-mean)/jax.numpy.sqrt(variance + eps)
    
    return params['gamma'] * x_nomalized + params['beta']


def transformer_block(params, x):
    x = x + multihead_attention(params, layer_norm(params['ln1'], x))
    x = x + ffn(params, layer_norm(params['ln2'], x))

    return x

def forward(params, inputs):
    # convert into embedding
    x = embed(params, inputs)
    # apply multiple blocks of multi-head attention
    for block_params in params['blocks']:
        x = transformer_block(block_params, x)
    # map to vocabulary
    return x @ params['W_o']


def loss_fn(params, inputs, outputs):
    logits = forward(params, inputs)
    return jax.numpy.mean(optax.softmax_cross_entropy_with_integer_labels(logits, outputs))


TRAINING_STEPS = 10000
def train(data, params, optimizer, optimizer_state, rand_key):
    @jax.jit
    def train_step(params, optimizer_state, inputs, outputs):
        loss, gradients = jax.value_and_grad(loss_fn)(params, inputs, outputs)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss


    for i in range(TRAINING_STEPS):
        rand_key, subkey = jax.random.split(rand_key)
        inputs, outputs = get_batch(data, subkey)
        
        params, optimizer_state, loss = train_step(params, optimizer_state, inputs, outputs)

        if i % 100 == 0:
            print(f"loss at {i} == {loss}")

    return params


def generate(params, prompt, length, rand_key, encode, decode):
    inputs = encode(prompt)
    for i in range(length):
        # inputs is just an array we need to convert it int (1, CONTEXT_LENGTH)

        # start with array of context length size with all zeros

        padded = jax.numpy.zeros(CONTEXT_LENGTH, dtype=jax.numpy.int32)

        # the token length could exceed the context length upon generation so 
        # remove the intial tokens
        tokens_to_include = jax.numpy.array(inputs[-CONTEXT_LENGTH:])

        # now we could also have the case where the input tokens are less
        # than the context size
        padded = padded.at[:len(tokens_to_include)].set(tokens_to_include)

        logits = forward(params, padded[None, :]) # this add the extra batch dimension
        
        predictions = logits[0, len(tokens_to_include) - 1]

        rand_key, subkey = jax.random.split(rand_key)

        prdiction = jax.random.categorical(subkey, predictions / 0.8)

        inputs.append(int(prdiction))

    print(decode(inputs))

def print_model_size(params):
    totals = params['token_embedding'].size

    totals += params['positional_embedding'].size

    for block in params['blocks']:
        totals += block['W_q'].size
        totals += block['W_k'].size
        totals += block['W_v'].size
        totals += block['W_o'].size
        totals += block['W1'].size
        totals += block['b1'].size
        totals += block['W2'].size
        totals += block['b2'].size
        totals += block['ln1']['gamma'].size
        totals += block['ln1']['beta'].size
        totals += block['ln2']['gamma'].size
        totals += block['ln2']['beta'].size

    totals += params['W_o'].size

    print(f"total model params: {totals}")

def main():
    parser = argparse.ArgumentParser(
        description="A tiny model"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--inference', action='store_true')
    parser.add_argument('--input', type=str, default='./input.txt', help='Path to input training data file')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--length', type=int, default=100)

    args = parser.parse_args()

    if args.train and args.prompt:
        parser.error("--train doesn't accept any other arguments")

    if args.inference and not args.prompt:
        parser.error("--inference requires --prompt")


    if args.train:
        text, encode, decode, char_to_encoding, encoding_to_char = load_data(args.input)
        encoded_data_numpy = jax.numpy.array(encode(text))

        params = init_params(rand_key)

        print_model_size(params)

        optimizer = optax.adam(learning_rate=3e-4)
        optimizer_state = optimizer.init(params)

        params = train(encoded_data_numpy, params, optimizer, optimizer_state, rand_key)

        with open("./params.pkl", "wb") as f:
            pickle.dump({'params': params, 'char_to_encoding': char_to_encoding, 'encoding_to_char': encoding_to_char}, f)

    elif args.inference:
        with open("./params.pkl", "rb") as f:
            checkpoint = pickle.load(f)

        params = checkpoint['params']
        char_to_encoding = checkpoint['char_to_encoding']
        encoding_to_char = checkpoint['encoding_to_char']

        encode = lambda text: [char_to_encoding[c] for c in text]
        decode = lambda encoding: "".join([encoding_to_char[i] for i in encoding])

        print_model_size(params)

        generate(params, args.prompt, args.length, rand_key, encode, decode)

if __name__ == "__main__":
    main()
