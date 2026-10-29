"""
mycoder - your own offline AI coding assistant.

It runs a real open-source coding model (Qwen2.5-Coder by default) on your own
computer through Ollama. No internet needed after the model is downloaded.
Only uses the Python standard library.

Setup (once):
  1. Install Ollama:            https://ollama.com/download
  2. Download a coding model:   ollama pull qwen2.5-coder:7b
  3. (optional) give it your own name and personality:
                                ollama create mycoder -f Modelfile

Usage:
  python coder.py chat                     # talk to it in the terminal
  python coder.py ask "write a python function that reverses a linked list"
  python coder.py review app.py            # find bugs in a file
  python coder.py fix app.py               # write a fixed copy: app.fixed.py
  python coder.py explain app.py           # explain what the code does
  python coder.py web                      # chat in your browser at http://localhost:8000
  python coder.py models                   # list models installed in Ollama

Pick a model with --model, e.g. --model qwen2.5-coder:1.5b on a weak laptop.
"""
import argparse, json, os, re, sys, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("MYCODER_MODEL", "mycoder")
FALLBACK_MODEL = "qwen2.5-coder:7b"

SYSTEM = (
    "You are mycoder, an expert programming assistant. You write correct, clean, "
    "well-structured code, explain your reasoning briefly, and point out bugs, edge "
    "cases and security problems honestly. When you are not sure, say so."
)

REVIEW_PROMPT = """Review the following file for bugs.

For each problem give:
- the line number(s)
- what is wrong and why it matters (crash, wrong result, security issue, edge case)
- how to fix it, with a short code snippet

Ignore pure style nitpicks. If you find no real bugs, say so.

File: {name}
```{lang}
{code}
```"""

FIX_PROMPT = """Fix every bug in the following file. Keep the same behaviour, structure and
style wherever the code is already correct.

Reply with exactly two parts:
1. A short bullet list of what you changed and why.
2. The COMPLETE fixed file in a single ```{lang} code block.

File: {name}
```{lang}
{code}
```"""

EXPLAIN_PROMPT = """Explain what this file does, for a developer who has never seen it.
Start with a 2-3 sentence summary, then walk through the main parts.

File: {name}
```{lang}
{code}
```"""

LANGS = {".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "jsx",
         ".tsx": "tsx", ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp",
         ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
         ".kt": "kotlin", ".swift": "swift", ".sh": "bash", ".sql": "sql",
         ".html": "html", ".css": "css", ".dart": "dart"}


# ----------------------------------------------------------------------------- Ollama client
def _post(path, payload, stream):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.URLError as e:
        sys.exit(f"Can't reach Ollama at {OLLAMA} ({e}).\n"
                 "Install it from https://ollama.com/download and make sure it is running.")


def installed_models():
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=10) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except urllib.error.URLError:
        return None


def resolve_model(name):
    models = installed_models()
    if models is None:
        sys.exit(f"Can't reach Ollama at {OLLAMA}. Is it installed and running? "
                 "https://ollama.com/download")
    def has(n):
        return any(m == n or m.split(":")[0] == n for m in models)
    if has(name):
        return name
    if name == DEFAULT_MODEL and has(FALLBACK_MODEL):
        return FALLBACK_MODEL
    coders = [m for m in models if "coder" in m]
    if coders:
        print(f"[model '{name}' not found, using '{coders[0]}']", file=sys.stderr)
        return coders[0]
    sys.exit(f"Model '{name}' is not installed. Run:  ollama pull {FALLBACK_MODEL}")


def chat_stream(model, messages, temperature=0.2, on_token=None):
    """Send a conversation to the model and stream the reply. Returns the full text."""
    payload = {"model": model, "messages": messages, "stream": True,
               "options": {"temperature": temperature, "num_ctx": 8192}}
    out = []
    with _post("/api/chat", payload, stream=True) as r:
        for line in r:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                sys.exit("Ollama error: " + chunk["error"])
            tok = chunk.get("message", {}).get("content", "")
            if tok:
                out.append(tok)
                if on_token:
                    on_token(tok)
            if chunk.get("done"):
                break
    return "".join(out)


def print_token(t):
    sys.stdout.write(t)
    sys.stdout.flush()


# ----------------------------------------------------------------------------- helpers
def read_code(path):
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    with open(path, encoding="utf-8", errors="replace") as f:
        code = f.read()
    if len(code) > 60_000:
        print("[warning: big file, only the first 60,000 characters are sent]", file=sys.stderr)
        code = code[:60_000]
    numbered = "\n".join(f"{i + 1:4d}| {l}" for i, l in enumerate(code.splitlines()))
    lang = LANGS.get(os.path.splitext(path)[1].lower(), "")
    return code, numbered, lang


def last_code_block(text):
    blocks = re.findall(r"```[\w+#.-]*\n(.*?)```", text, flags=re.S)
    return max(blocks, key=len) if blocks else None


def one_shot(model, prompt):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = chat_stream(model, msgs, on_token=print_token)
    print()
    return text


# ----------------------------------------------------------------------------- commands
def cmd_chat(a):
    model = resolve_model(a.model)
    print(f"mycoder ({model}) - offline. Commands: /file PATH (attach a file), /clear, /quit\n")
    msgs = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/clear":
            msgs = msgs[:1]; print("[conversation cleared]\n"); continue
        if q.startswith("/file "):
            path = q[6:].strip()
            code, _, lang = read_code(path)
            q = input("question about the file> ").strip() or "Review this file for bugs."
            q = f"{q}\n\nFile: {path}\n```{lang}\n{code}\n```"
        msgs.append({"role": "user", "content": q})
        sys.stdout.write("mycoder> ")
        reply = chat_stream(model, msgs, on_token=print_token)
        print("\n")
        msgs.append({"role": "assistant", "content": reply})


def cmd_ask(a):
    one_shot(resolve_model(a.model), " ".join(a.question))


def cmd_review(a):
    _, numbered, lang = read_code(a.file)
    one_shot(resolve_model(a.model), REVIEW_PROMPT.format(name=a.file, lang=lang, code=numbered))


def cmd_explain(a):
    code, _, lang = read_code(a.file)
    one_shot(resolve_model(a.model), EXPLAIN_PROMPT.format(name=a.file, lang=lang, code=code))


def cmd_fix(a):
    code, _, lang = read_code(a.file)
    text = one_shot(resolve_model(a.model), FIX_PROMPT.format(name=a.file, lang=lang, code=code))
    fixed = last_code_block(text)
    if not fixed:
        sys.exit("\n[the model did not return a code block, nothing written]")
    root, ext = os.path.splitext(a.file)
    out = a.out or f"{root}.fixed{ext}"
    with open(out, "w", encoding="utf-8") as f:
        f.write(fixed if fixed.endswith("\n") else fixed + "\n")
    print(f"\n[fixed file written to {out} - your original was not changed]")


def cmd_models(a):
    models = installed_models()
    if models is None:
        sys.exit(f"Can't reach Ollama at {OLLAMA}.")
    print("\n".join(models) if models else "No models yet. Try: ollama pull qwen2.5-coder:7b")


WEB_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>mycoder</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#e6e8ee;--dim:#8a91a3;--acc:#7aa2f7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}
header b{font-size:17px}header span{color:var(--dim);font-size:13px}
#log{flex:1;overflow-y:auto;padding:16px;max-width:900px;width:100%;margin:0 auto}
.m{margin:0 0 18px;white-space:pre-wrap;word-wrap:break-word}.u{color:var(--acc)}.u:before{content:"you  ";color:var(--dim)}
.a:before{content:"mycoder  ";color:var(--dim)}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;white-space:pre}
form{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--line);max-width:900px;width:100%;margin:0 auto}
textarea{flex:1;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:10px;font:14px ui-monospace,monospace;resize:vertical;min-height:48px}
button{background:var(--acc);color:#0b0d12;border:0;border-radius:8px;padding:0 18px;font-weight:600;cursor:pointer}
button:disabled{opacity:.5}
</style></head><body>
<header><b>mycoder</b><span id="model"></span></header>
<div id="log"></div>
<form id="f"><textarea id="q" placeholder="Ask for code, or paste code and ask me to find bugs. Enter to send, Shift+Enter for a new line."></textarea><button id="b">Send</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),b=document.getElementById('b');
let msgs=[];
fetch('/model').then(r=>r.text()).then(t=>document.getElementById('model').textContent=t+' · offline');
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function render(t){return esc(t).replace(/```[\w+#.-]*\n?([\s\S]*?)(```|$)/g,(m,c)=>'<pre>'+c+'</pre>')}
function add(cls,t){const d=document.createElement('div');d.className='m '+cls;d.innerHTML=render(t);log.appendChild(d);log.scrollTop=1e9;return d}
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit()}});
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const text=q.value.trim();if(!text)return;q.value='';b.disabled=true;
  add('u',text);msgs.push({role:'user',content:text});const d=add('a','…');let out='';
  try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:msgs})});
    const rd=r.body.getReader(),dec=new TextDecoder();
    while(true){const {done,value}=await rd.read();if(done)break;out+=dec.decode(value,{stream:true});d.innerHTML=render(out);log.scrollTop=1e9}
  }catch(err){out='[error: '+err+']';d.textContent=out}
  msgs.push({role:'assistant',content:out});b.disabled=false;q.focus();
};
</script></body></html>"""


def cmd_web(a):
    model = resolve_model(a.model)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = (model if self.path == "/model" else WEB_PAGE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8" if self.path == "/model"
                             else "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/chat":
                self.send_error(404); return
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            msgs = [{"role": "system", "content": SYSTEM}] + data.get("messages", [])[-30:]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()

            def send(tok):
                self.wfile.write(tok.encode()); self.wfile.flush()
            try:
                chat_stream(model, msgs, on_token=send)
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"mycoder web UI: http://localhost:{a.port}  (model: {model}, Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()


def main():
    ap = argparse.ArgumentParser(description="mycoder - offline AI coding assistant")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat")
    s = sub.add_parser("ask"); s.add_argument("question", nargs="+")
    for name in ("review", "explain"):
        s = sub.add_parser(name); s.add_argument("file")
    s = sub.add_parser("fix"); s.add_argument("file"); s.add_argument("--out")
    s = sub.add_parser("web"); s.add_argument("--port", type=int, default=8000)
    sub.add_parser("models")
    a = ap.parse_args()
    {"chat": cmd_chat, "ask": cmd_ask, "review": cmd_review, "explain": cmd_explain,
     "fix": cmd_fix, "web": cmd_web, "models": cmd_models}[a.cmd](a)


if __name__ == "__main__":
    main()
