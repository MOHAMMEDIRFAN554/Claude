"""
Fine-tune an open-source coding model on your own data with LoRA.

LoRA trains a small set of extra weights (a few MB) on top of the base model,
so it fits on one GPU - including the free T4 GPU on Google Colab.

  pip install -r requirements.txt
  python train_lora.py --data data/train.jsonl
  python train_lora.py --data data/train.jsonl --model Qwen/Qwen2.5-Coder-7B-Instruct --4bit

Data format: one JSON object per line, in chat format:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
The model only learns from the assistant replies; the user turns are context.
"""
import argparse, json, math, os, random

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SYSTEM = ("You are mycoder, an expert programming assistant. You write correct, clean, "
          "well-structured code, explain your reasoning briefly, and point out bugs, edge "
          "cases and security problems honestly. When you are not sure, say so.")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row["messages"]
            if not any(m["role"] == "assistant" for m in msgs):
                raise ValueError(f"{path}:{n} has no assistant message")
            if msgs[0]["role"] != "system":
                msgs = [{"role": "system", "content": SYSTEM}] + msgs
            rows.append(msgs)
    return rows


class ChatDataset(Dataset):
    """Tokenizes conversations; labels are -100 everywhere except assistant replies."""

    def __init__(self, convos, tok, max_len):
        self.items = []
        skipped = 0
        for msgs in convos:
            ids, labels = [], []
            for i, m in enumerate(msgs):
                # text of the conversation up to and including message i
                upto = tok.apply_chat_template(msgs[: i + 1], tokenize=False)
                prev = tok.apply_chat_template(msgs[:i], tokenize=False) if i else ""
                piece = tok(upto[len(prev):], add_special_tokens=False)["input_ids"]
                ids += piece
                labels += piece if m["role"] == "assistant" else [-100] * len(piece)
            if len(ids) > max_len:
                ids, labels = ids[:max_len], labels[:max_len]
            if all(l == -100 for l in labels):
                skipped += 1
                continue
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
        if skipped:
            print(f"skipped {skipped} examples whose answer was cut off by --max_len")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    help="base model from huggingface.co (1.5B fits a free Colab T4; 7B needs --4bit)")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--eval_data", default=None, help="optional held-out jsonl")
    ap.add_argument("--out", default="out/mycoder-lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="load base model in 4-bit (QLoRA) to save GPU memory")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        print("WARNING: no NVIDIA GPU found. This will be extremely slow. Use Google Colab "
              "(Runtime > Change runtime type > T4 GPU) - see colab.ipynb.")

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    kw = {"torch_dtype": dtype}
    if a.four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    if torch.cuda.is_available():
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if a.four_bit:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train_ds = ChatDataset(load_jsonl(a.data), tok, a.max_len)
    eval_ds = ChatDataset(load_jsonl(a.eval_data), tok, a.max_len) if a.eval_data else None
    print(f"training examples: {len(train_ds)}" + (f" | eval: {len(eval_ds)}" if eval_ds else ""))
    if len(train_ds) == 0:
        raise SystemExit("no usable training examples")

    steps_per_epoch = math.ceil(len(train_ds) / (a.batch_size * a.grad_accum))
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, lr_scheduler_type="cosine", warmup_ratio=0.05,
        logging_steps=max(1, steps_per_epoch // 10), save_strategy="epoch", save_total_limit=2,
        eval_strategy="epoch" if eval_ds else "no",
        bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True, report_to="none", seed=a.seed,
        optim="paged_adamw_8bit" if a.four_bit else "adamw_torch",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "base_model.txt"), "w") as f:
        f.write(a.model)
    print(f"\nLoRA adapter saved to {a.out}\nNext: python export.py --lora {a.out}")


if __name__ == "__main__":
    main()
