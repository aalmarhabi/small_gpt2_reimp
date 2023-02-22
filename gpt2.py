from jax import numpy as jnp


'''

Gaussian Error Linear Units (GeLU)
  is an alternative to Rectified Linear Unit ReLU activation function. check out (https://arxiv.org/pdf/1606.08415.pdf)
  approximation of GeLU is 0.5 * x * (1 + tanh(SqaureRoot(2/pi) * (x + 0.044715 * (x)^3))) or x*SmallSigma(1.702x)

'''
def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))

'''

Softmax(x_i) = exp^(x_i) / Sigma_j (e^(x_j)). For large inputs softmax become unstable because of infinity and Nan cases.
  We can prevent large number by subtract x_i by max(x). So:
  Softmax(x_i) = exp^(x_i - max(x)) / Sigma_j (e^(x_j - max(x)))

'''

def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

    
'''

Layer Normalization
  LN(x) = _y * ( (x - _mu) / (SquareRoot(_sigma^2)) ) + _Lbeta
  _y => gamma ; _lbeta => beta ; _mu => mean of x ; _sigma^2 => variance
  Standardizes the values to mean 0 and variance 1
  In Transformer Layer (features) norm is used more than Batch norm.
  https://tungmphung.com/deep-learning-normalization-methods/ 

'''

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(variance + eps) # normalize x to mean=0 and var=1
    return g * x + b # scale and offset with gamma/beta params

'''

Linear
  projections as a standard matrix multiplication + bias

'''
def linear(x, w, b): # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b  # @ numpy matrix multiplication

'''

position-wise feed forward network (ffn) multi-layer perception with 2 layers

'''
def ffn(x, c_fc, c_proj):
    # project up
    a = gelu(linear(x, **c_fc))   # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, n_embd]  -> [n_seq, n_embd]

    return x 

'''

attention
  scaled dot product attention (check attention is all you need). 

'''

def attention(q, k, v, mask):
    return softmax(q @ k.T / jnp.sqrt(q.shape[-1]) + mask) @ v


'''

Multi-head attention (mha)
  Improve model performing by applying n_head separate attention.
   1 - Split q, k, v into n_head
   2 - Compute attention for each head
   3 - Merge output of each head
  Note this reduces dimension n_embd to n_embd/n_head

'''

def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd]  ->  [n_seq, 3*n_embd]

    # split into qkv
    qkv = jnp.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: jnp.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # casual mask to hide future inputs from being attended to
    causal_mask = (1 - jnp.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = jnp.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head]  -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x

    


'''

Decoder block or transformer_block use two important sub-layers
  1 - multi-head casual self-attention (mha)
  2 - position--wise feed forward neural network (ffn)

'''
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd]  -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd]  -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd]  -> [n_seq, n_embd]

    return x

'''

Generative Pre-trained Transformer model 2
  wte is token embedding [n_vocab, n_embd] matrix (lookup table)
  wpe is positional embedding [nn_ctx, n_embd] matrix
  combined token and positional embedding wte + wpe
  blocks is stack of n_layer transformer decoder blocks ,e.g., GPT-3 has 96 layers

'''

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq]  ->  [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[jnp.array(inputs)] + wpe[jnp.array(range(len(inputs)))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head) # [n_seq, n_embd]  -> [n_seq, n_embd]

    # projection to vocabulary
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd]  ->  [n_seq, n_vocab]


'''

Generate
  Using auto-regressive to generate full sentences iteratively. For each iteration we append predication back to input
tqdm (https://tqdm.github.io/)
  library that make your loops show a smart progress meter

'''

def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"): # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head) # model forward pass
        next_id = jnp.argmax(logits[-1])  # greedy sampling for more randomness of auto-regressive
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]   # only return generated ids


'''

main
  load the openai encoder (tokenizer)
  generate text ids by the encoder
  convert text ids into string

'''

def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    # utils.py
    from utils import load_encoder_hparams_and_params

    # load encoder, hyperparamters, and params from official openai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the byte pair encoding (BPE) tokenizer
    input_ids = encoder.encode(prompt)

    # make sure the max sequence length of model is not off limit
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"] # if false raise alert

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    print(output_ids)
    output_text = encoder.decode(output_ids)

    return output_text


'''

Use Python-Fire for automatically generating command line interfaces (CLIs)
Fire convert main function parameters into CLIs
  python gpt2.py --prompt="write prompt here" --n_tokens_to_generate=40 --model_size="124M" --models_dir="models"


'''

if __name__ == "__main__":
    import fire

    fire.Fire(main)

    
