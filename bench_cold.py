"""True-cold swap benchmark (paper Sec 5.1). Per rep: 12GB cache turnover
(sequential read of two 6GB eviction files, created by evict_probe.py),
then timed checkpoint load + 1-token generation. 12 reps x 3 models.
Also records warm raw-read baselines to decompose load = IO + CPU overhead.
"""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import gc, json, time
from time import perf_counter
import statistics

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

PROMPT = "Why is my Linux server consuming 100% CPU?"
MODELS = ["distilgpt2", "gpt2-medium", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
REPS = 12
TMP = os.environ.get("TEMP", r"C:\Users\MUKESH\AppData\Local\Temp")
EV = [os.path.join(TMP, "evict1.bin"), os.path.join(TMP, "evict2.bin")]
LOG = "bench_cold.log"

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def raw_read(path, chunk=4 * 2**20):
    t0 = perf_counter()
    with open(path, "rb") as f:
        while f.read(chunk):
            pass
    return perf_counter() - t0

def weights_path(mid):
    snap = snapshot_download(mid, allow_patterns=["*.safetensors"])
    for f in os.listdir(snap):
        if f.endswith(".safetensors"):
            return os.path.join(snap, f)
    raise RuntimeError("no weights in " + snap)

def summarize(v):
    return dict(n=len(v), mean=round(statistics.mean(v), 3),
                median=round(statistics.median(v), 3),
                stdev=round(statistics.stdev(v), 3) if len(v) > 1 else 0.0,
                min=round(min(v), 3), max=round(max(v), 3))

def main():
    open(LOG, "w").write("cold bench start\n")
    out = {"reps": REPS, "models": {}}
    for mid in MODELS:
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        wp = weights_path(mid)
        wsize = os.path.getsize(wp)
        # warm raw-read baseline (file currently cached from prior use)
        warm_reads = [raw_read(wp) for _ in range(3)]
        loads, toks, tots = [], [], []
        for i in range(REPS):
            for p in EV:  # turnover: force eviction of model bytes
                raw_read(p)
            t0 = perf_counter()
            model = AutoModelForCausalLM.from_pretrained(mid)
            model.eval()
            t1 = perf_counter()
            ids = tok(PROMPT, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                model.generate(ids, max_new_tokens=1, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            t2 = perf_counter()
            loads.append(t1 - t0); toks.append(t2 - t1); tots.append(t2 - t0)
            del model
            gc.collect()
            log(f"{mid} cold rep {i+1}/{REPS} load={t1-t0:.2f}s tok1={t2-t1:.2f}s")
        out["models"][mid] = {
            "weights_bytes": wsize,
            "warm_raw_read_s": summarize(warm_reads),
            "cold": {"load_s": summarize(loads),
                     "first_token_s": summarize(toks),
                     "total_s": summarize(tots)},
        }
        del tok
        gc.collect()
    with open("bench_cold.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log("COLD ALL DONE")

if __name__ == "__main__":
    main()
