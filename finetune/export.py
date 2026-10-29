"""
Merge your LoRA adapter into the base model and convert it to GGUF for Ollama,
so your fine-tuned model runs offline with coder.py.

  python export.py --lora out/mycoder-lora
  ollama create mycoder -f out/Modelfile
  python ../coder/coder.py chat

Needs llama.cpp for the GGUF conversion; this script clones it if missing.
"""
import argparse, os, subprocess, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM = ("You are mycoder, an expert programming assistant. You write correct, clean, "
          "well-structured code, explain your reasoning briefly, and point out bugs, edge "
          "cases and security problems honestly. When you are not sure, say so.")


def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="out/mycoder-lora")
    ap.add_argument("--base", default=None, help="base model (default: read from the adapter folder)")
    ap.add_argument("--merged", default="out/mycoder-merged")
    ap.add_argument("--gguf", default="out/mycoder.gguf")
    ap.add_argument("--quant", default="q8_0", choices=["f16", "bf16", "q8_0"],
                    help="q8_0 is about half the size of f16 with almost no quality loss")
    ap.add_argument("--llama_cpp", default="llama.cpp")
    a = ap.parse_args()

    base = a.base or open(os.path.join(a.lora, "base_model.txt")).read().strip()
    print(f"merging {a.lora} into {base} ...")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, a.lora).merge_and_unload()
    model.save_pretrained(a.merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.lora).save_pretrained(a.merged)
    print(f"merged model saved to {a.merged}")

    if not os.path.isdir(a.llama_cpp):
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", a.llama_cpp])
        run([sys.executable, "-m", "pip", "install", "-q", "-r",
             os.path.join(a.llama_cpp, "requirements", "requirements-convert_hf_to_gguf.txt")])
    run([sys.executable, os.path.join(a.llama_cpp, "convert_hf_to_gguf.py"), a.merged,
         "--outfile", a.gguf, "--outtype", a.quant])

    mf = os.path.join(os.path.dirname(a.gguf) or ".", "Modelfile")
    with open(mf, "w") as f:
        f.write(f"FROM ./{os.path.basename(a.gguf)}\n\nPARAMETER temperature 0.2\n"
                f"PARAMETER num_ctx 8192\n\nSYSTEM \"\"\"{SYSTEM}\"\"\"\n")
    print(f"\nDone! GGUF: {a.gguf}\nNow run:\n  ollama create mycoder -f {mf}\n"
          f"  python ../coder/coder.py chat")


if __name__ == "__main__":
    main()
