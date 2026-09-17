"""Download the three benchmark models to local HF cache. Logs progress to dl.log."""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"  # XET endpoint 404s from this network; use plain HTTP
import time, traceback
from huggingface_hub import snapshot_download

MODELS = ["distilgpt2", "gpt2-medium", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
LOG = r"E:\Documents\Default Project\New folder\dl.log"

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

open(LOG, "w").write("download start\n")
for m in MODELS:
    try:
        log(f"START {m}")
        # distilgpt2/gpt2-medium: prefer safetensors weight only to save time/space
        allow = ["*.json", "*.txt", "*.safetensors", "*.model", "tokenizer*"]
        p = snapshot_download(m, allow_patterns=allow)
        log(f"DONE {m} -> {p}")
    except Exception:
        log(f"FAIL {m}\n{traceback.format_exc()}")
log("ALL DONE")
