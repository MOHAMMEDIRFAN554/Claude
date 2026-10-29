# mygpt

A tiny GPT (the same kind of transformer model behind ChatGPT and Claude, just much, much smaller) written from scratch in pure Python + NumPy. No PyTorch, no GPU needed.

## What's inside

- `gpt.py`: the whole model in one file
  - character-level tokenizer
  - token + position embeddings
  - transformer blocks: LayerNorm, multi-head causal self-attention, MLP with GELU, residual connections
  - hand-written backpropagation (verified with a gradient check)
  - AdamW optimizer with warmup and cosine learning-rate decay
  - text generation with temperature and top-k sampling
- `input.txt`: short original stories used as training data
- `mygpt.npz`: trained weights (about 350K parameters), ready to use

## Quick start

```bash
pip install numpy

# check the backprop math is correct
python gpt.py gradcheck

# train (takes a few minutes on a laptop CPU)
python gpt.py train --data input.txt --steps 2500

# generate text
python gpt.py generate --prompt "Once upon a time" --length 400

# interactive mode
python gpt.py chat
```

## Make it smarter

- **More data:** replace `input.txt` with any large text file (books, articles, your own writing). More text = better model. The "Tiny Shakespeare" dataset (1 MB) is a classic choice.
- **Bigger model:** `--n_layer 6 --n_embd 192 --n_head 6 --block_size 128` (slower, but learns more).
- **Train longer:** `--steps 10000`.

## How it compares to Claude

Same core idea (predict the next token with a transformer), but Claude has hundreds of billions of parameters instead of 350 thousand, is trained on trillions of words, uses smarter tokenizers, and is then fine-tuned with human feedback to follow instructions and chat. This project is the first rung of that ladder.
