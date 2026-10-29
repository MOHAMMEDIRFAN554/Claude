# My AI

Three projects, from "how does a GPT work" to "an AI that writes code and finds bugs, offline".

| Folder | What it is | Needs |
|---|---|---|
| [`mygpt/`](mygpt) | A tiny GPT written from scratch in NumPy. Learn how transformers work. | Any computer |
| [`coder/`](coder) | **mycoder**: an offline AI coding assistant. Writes code, reviews files for bugs, fixes them. | Ollama + 8 GB RAM |
| [`finetune/`](finetune) | Train mycoder on your own code with LoRA. Runs free on Google Colab. | Free Colab GPU |

---

## 1. mycoder: offline coding assistant

Runs [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct), one of the strongest open-source coding models, on your own computer. No internet after setup, your code never leaves your machine.

**Setup (once)**
1. Install Ollama: https://ollama.com/download
2. Download the model (about 4.7 GB):
   ```
   ollama pull qwen2.5-coder:7b
   ```
   Laptop with 8 GB RAM or less? Use `qwen2.5-coder:1.5b` instead (1 GB, faster, a bit less smart).
3. Give it your own name and personality:
   ```
   cd coder
   ollama create mycoder -f Modelfile
   ```

**Use it**
```
python coder.py chat                  # chat in the terminal (/file PATH to attach a file)
python coder.py web                   # chat in your browser: http://localhost:8000
python coder.py ask "write a FastAPI endpoint that uploads a file"
python coder.py review app.py         # find bugs, with line numbers and fixes
python coder.py fix app.py            # writes app.fixed.py, original untouched
python coder.py explain app.py        # explain what a file does
```
Only needs Python 3.8+, no packages to install.

| Your computer | Model to use |
|---|---|
| 8 GB RAM, no GPU | `qwen2.5-coder:1.5b` or `3b` |
| 16 GB RAM | `qwen2.5-coder:7b` (recommended) |
| 32 GB RAM or a GPU with 16 GB+ | `qwen2.5-coder:14b` |
| GPU with 24 GB+ | `qwen2.5-coder:32b` (close to top commercial models) |

Change the model with `--model qwen2.5-coder:14b`, or edit the `FROM` line in `coder/Modelfile`.

## 2. Train it on your own code

Open [`finetune/colab.ipynb`](finetune/colab.ipynb) in Google Colab (File > Open notebook > GitHub, paste this repo's URL), switch to a T4 GPU, and run the cells. It:

1. turns your projects into training examples (`make_dataset.py`): code completion, explanations, and bug-finding exercises made by injecting realistic bugs into your own code,
2. fine-tunes the model with LoRA (`train_lora.py`),
3. exports it to a `mycoder.gguf` file you download and run offline with Ollama (`export.py`).

Best results come from adding your own hand-written examples to `finetune/data/example.jsonl` (one JSON line each, see the file for the format). A few hundred good examples beat thousands of auto-generated ones.

Have an NVIDIA GPU at home? Run the same scripts locally:
```
cd finetune
pip install -r requirements.txt
python make_dataset.py ~/my_projects --out data/train.jsonl
python train_lora.py --data data/train.jsonl --eval_data data/train.eval.jsonl
python export.py --lora out/mycoder-lora
ollama create mycoder -f out/Modelfile
```

## 3. mygpt: a GPT from scratch

```
cd mygpt
pip install numpy
python gpt.py chat
```
See [`mygpt/README.md`](mygpt/README.md). This one is for learning: every part of a transformer, including backpropagation, written by hand in ~400 lines.

## Honest expectations

- **mycoder** is genuinely useful for writing code, explaining it, and catching bugs. The 7B model is roughly at the level of commercial assistants from 2023; 32B is much closer to today's best. It will still make mistakes, so test what it writes.
- **Fine-tuning** teaches the model your style, libraries and common mistakes. It does not make a 1.5B model as smart as a 32B one; model size matters most.
- **mygpt** is trained on ~10 KB of stories, so it mostly remixes what it memorized. Give it a few MB of text and train longer for more original output.
