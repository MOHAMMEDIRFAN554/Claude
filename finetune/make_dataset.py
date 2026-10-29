"""
Build a training dataset from your own code.

  python make_dataset.py path/to/your/projects --out data/train.jsonl

It creates three kinds of examples from every source file it finds:
  1. explain   - "what does this code do?"  -> a docstring/comment-based summary (when the file has one)
  2. complete  - "finish this code"         -> the real next part of your file
  3. bug-fix   - code with a small injected bug -> "here is the bug, here is the fix"

The bug-fix examples teach the model to spot common mistakes (off-by-one,
wrong comparison, swapped operator, missing return...) in YOUR style of code.
You can also write examples by hand in the same format - those help the most.
Merge any extra .jsonl files in with --include.
"""
import argparse, json, os, random, re

EXTS = {".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "jsx", ".tsx": "tsx",
        ".java": "java", ".c": "c", ".cpp": "cpp", ".cs": "csharp", ".go": "go", ".rs": "rust",
        ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".swift": "swift", ".dart": "dart"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", "target",
             ".next", ".idea", ".vscode", "vendor"}

# (pattern, replacement, description of the bug that was injected)
MUTATIONS = [
    (r"(?<![<>=!])<=(?!=)", "<", "an off-by-one error: `<=` was changed to `<`, so the last value is skipped"),
    (r"(?<![<>=!])<(?![<=])", "<=", "an off-by-one error: `<` was changed to `<=`, so it goes one step too far"),
    (r"(?<![=!<>])==(?!=)", "!=", "an inverted condition: `==` was changed to `!=`"),
    (r"!=", "==", "an inverted condition: `!=` was changed to `==`"),
    (r"(?<=\w) \+ (?=\w)", " - ", "a wrong operator: `+` was changed to `-`"),
    (r"(?<=\w) - (?=\w)", " + ", "a wrong operator: `-` was changed to `+`"),
    (r"\band\b", "or", "a logic error: `and` was changed to `or`"),
    (r"&&", "||", "a logic error: `&&` was changed to `||`"),
    (r"\bTrue\b", "False", "a wrong boolean: `True` was changed to `False`"),
    (r"\btrue\b", "false", "a wrong boolean: `true` was changed to `false`"),
    (r"\[0\]", "[1]", "a wrong index: `[0]` was changed to `[1]`"),
    (r"\[-1\]", "[0]", "a wrong index: `[-1]` (last item) was changed to `[0]` (first item)"),
]


def find_files(roots, max_bytes):
    for root in roots:
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                p = os.path.join(d, fn)
                if ext in EXTS and 200 <= os.path.getsize(p) <= max_bytes:
                    yield p, EXTS[ext]


def chunks(lines, size=60):
    for i in range(0, len(lines), size):
        part = lines[i:i + size]
        if len("".join(part).strip()) > 150:
            yield i, part


def msg(user, assistant):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def make_complete(path, lang, lines, rng):
    out = []
    for start, part in chunks(lines, 50):
        cut = rng.randint(len(part) // 3, max(len(part) // 3, 2 * len(part) // 3))
        head, tail = "".join(part[:cut]), "".join(part[cut:])
        if head.strip() and tail.strip():
            out.append(msg(f"Continue this {lang} code from `{os.path.basename(path)}`:\n```{lang}\n{head}```",
                           f"```{lang}\n{tail}```"))
    return out


def make_bugfix(path, lang, lines, rng):
    out = []
    for start, part in chunks(lines, 40):
        code_lines = [i for i, l in enumerate(part) if l.strip() and not l.strip().startswith(("#", "//", "*", "/*"))]
        rng.shuffle(code_lines)
        for i in code_lines[:15]:
            line = part[i]
            options = [(p, r, why) for p, r, why in MUTATIONS if re.search(p, line)]
            if not options:
                continue
            p, r, why = rng.choice(options)
            bad = re.sub(p, r, line, count=1)
            if bad == line:
                continue
            buggy = part[:i] + [bad] + part[i + 1:]
            ln = i + 1
            answer = (f"There is a bug on line {ln}: {why}.\n\n"
                      f"Buggy line:\n```{lang}\n{bad.rstrip()}\n```\n"
                      f"Fixed line:\n```{lang}\n{line.rstrip()}\n```")
            numbered = "".join(f"{j + 1:3d}| {l}" for j, l in enumerate(buggy))
            out.append(msg(f"Find the bug in this {lang} code:\n```{lang}\n{numbered}```", answer))
            break
    return out


def make_explain(path, lang, text):
    m = re.match(r'\s*(?:"""|\'\'\')(.+?)(?:"""|\'\'\')', text, re.S) if lang == "python" else \
        re.match(r"\s*/\*\*?(.+?)\*/", text, re.S)
    if not m:
        return []
    doc = re.sub(r"^\s*\* ?", "", m.group(1).strip(), flags=re.M).strip()
    if len(doc) < 60:
        return []
    body = text[m.end():][:6000]
    return [msg(f"What does this {lang} file do?\n```{lang}\n{body}\n```", doc)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="folders with your code")
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--include", nargs="*", default=["data/example.jsonl"],
                    help="hand-written .jsonl files to mix in")
    ap.add_argument("--max_file_kb", type=int, default=200)
    ap.add_argument("--eval_fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    examples, n_files = [], 0
    for path, lang in find_files(a.roots, a.max_file_kb * 1024):
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        lines = text.splitlines(keepends=True)
        n_files += 1
        examples += make_explain(path, lang, text)
        examples += make_complete(path, lang, lines, rng)
        examples += make_bugfix(path, lang, lines, rng)

    for inc in a.include or []:
        if os.path.isfile(inc):
            with open(inc, encoding="utf-8") as f:
                hand = [json.loads(l) for l in f if l.strip()]
            examples += hand * 3            # hand-written examples are worth more
            print(f"included {len(hand)} hand-written examples from {inc} (x3)")

    rng.shuffle(examples)
    n_eval = int(len(examples) * a.eval_fraction)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    eval_path = a.out.replace(".jsonl", ".eval.jsonl")
    for path, rows in ((a.out, examples[n_eval:]), (eval_path, examples[:n_eval])):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"scanned {n_files} files -> {len(examples) - n_eval} train / {n_eval} eval examples")
    print(f"wrote {a.out} and {eval_path}")


if __name__ == "__main__":
    main()
