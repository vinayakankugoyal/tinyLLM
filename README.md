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

### Training

```bash
usage: tinyLLM.py [-h] (--train | --inference) [--input INPUT] [--prompt PROMPT] [--length LENGTH]

A tiny model

options:
  -h, --help       show this help message and exit
  --train
  --inference
  --input INPUT    Path to input training data file
  --prompt PROMPT
  --length LENGTH
```

Runs for 10,000 steps and saves weights to `params.pkl`.

### Inference

```bash
python tinyLLM.py --inference --prompt "First Citizen:"
```

Generates 100 tokens from the given prompt using the trained model.
