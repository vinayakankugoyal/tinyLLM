# tinyLLM

A minimal transformer implementation in JAX to understand how multi-head attention works.

## Overview

This is an educational implementation of a decoder-only transformer, similar in spirit to GPT. The model is trained on Shakespeare text and can generate new text in a similar style.

## Architecture

- 4 transformer blocks with multi-head attention (4 heads)
- 128-dimensional embeddings
- Context length of 128 tokens
- Character-level tokenization (65 vocabulary size)
- Layer normalization and feedforward networks
- ~800K parameters total

## Usage

```bash
usage: tinyLLM.py [-h] (--train | --inference) [--input INPUT] [--params PARAMS] [--prompt PROMPT] [--length LENGTH]

A tiny model

options:
  -h, --help       show this help message and exit
  --train          Train the model on input data
  --inference      Generate text using a trained model
  --input INPUT    Path to input training data file
  --params PARAMS  Path to model parameters file
  --prompt PROMPT  Starting text for generation (required for inference)
  --length LENGTH  Number of tokens to generate
```

### Training

```
python tinyLLM.py --train
```

Runs for 10,000 steps and saves weights to `params.pkl`.

### Inference

```bash
[nix-shell:~/projects/tinyLLM]$ python tinyLLM.py --inference --prompt "to be or not to be" --length 200
total model params: 824064
to be or not to be.

First Senator:
For I be you; fellow present in your highness,
I hear my friend count and clears,
I both prince, give me sooth and with our soul.

LADY CAPULET:
O holy wrong up yout to or as little

```

Generates tokens from the given prompt using the trained model.
