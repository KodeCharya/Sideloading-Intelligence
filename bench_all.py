"""Swap-latency + memory benchmark (paper Sec 5.1, Table 1).

For each of 3 real checkpoints spanning ~0.35/1.5/2.2 GB on disk:
  warm: file resident in OS page cache (2 untimed preloads, then N timed loads)
  cold: user-space cache displacement before every rep (8 GB buffer, touched
        page by page, then freed) -- approximation; no admin cache flush used.
Each rep times: checkpoint load (disk->RAM, from_pretrained on CPU) and
time-to-first-token (generate 1 token, greedy). Reports mean/median/stdev.
Also records process RSS at each stage and a Brain+N-skills residency test.
Appends JSONL per rep so partial results survive; final summary JSON at end.
"""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import gc, json, time, statistics
from time import perf_counter

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Why is my Linux server consuming 100% CPU?"
MODELS = [
    ("distilgpt2", "skill-small"),
    ("gpt2-medium", "skill-mid"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "brain"),
]
REPS = 12
LOG = "bench.log"
JSONL = "bench_reps.jsonl"

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def rss_mb():
    return psutil.Process().memory_info().rss / 1e6

def displace_cache(gb=8):
    n = gb * 2**30
    b = bytearray(n)
    for i in range(0, n, 4096):
        b[i] = (i >> 12) & 255
    del b
    gc.collect()

def timed_load_generate(mid, tok):
    t0 = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(mid)
    model.eval()
    t1 = perf_counter()
    ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        model.generate(ids, max_new_tokens=1, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    t2 = perf_counter()
    r = rss_mb()
    del model
    gc.collect()
    return (t1 - t0), (t2 - t1), (t2 - t0), r

def summarize(v):
    return dict(n=len(v), mean=round(statistics.mean(v), 3),
                median=round(statistics.median(v), 3),
                stdev=round(statistics.stdev(v), 3) if len(v) > 1 else 0.0,
                min=round(min(v), 3), max=round(max(v), 3))

def main():
    open(LOG, "w").write("bench start\n")
    open(JSONL, "w").write("")
    proc = psutil.Process()
    out = {"prompt": PROMPT, "reps": REPS,
           "torch": torch.__version__,
           "baseline_rss_mb": round(rss_mb(), 1),
           "models": {}}
    log(f"baseline RSS={out['baseline_rss_mb']} MB threads={torch.get_num_threads()}")
    for mid, tag in MODELS:
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = {}
        # warm: 2 untimed preloads then timed reps back-to-back
        for _ in range(2):
            timed_load_generate(mid, tok)
        W = [timed_load_generate(mid, tok) for _ in range(REPS)]
        log(f"{tag} WARM done n={REPS}")
        # cold: displace page cache before every rep
        C = []
        for i in range(REPS):
            displace_cache(8)
            C.append(timed_load_generate(mid, tok))
            with open(JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps({"model": mid, "cond": "cold", "rep": i,
                                    "load_s": round(C[-1][0], 3),
                                    "tok1_s": round(C[-1][1], 3),
                                    "total_s": round(C[-1][2], 3)}) + "\n")
            log(f"{tag} cold rep {i+1}/{REPS} load={C[-1][0]:.1f}s tok1={C[-1][1]:.1f}s")
        for i, w in enumerate(W):
            with open(JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps({"model": mid, "cond": "warm", "rep": i,
                                    "load_s": round(w[0], 3),
                                    "tok1_s": round(w[1], 3),
                                    "total_s": round(w[2], 3)}) + "\n")
        m["warm"] = {"load_s": summarize([w[0] for w in W]),
                     "first_token_s": summarize([w[1] for w in W]),
                     "total_s": summarize([w[2] for w in W]),
                     "rss_after_load_mb": round(statistics.mean([w[3] for w in W]), 1)}
        m["cold"] = {"load_s": summarize([c[0] for c in C]),
                     "first_token_s": summarize([c[1] for c in C]),
                     "total_s": summarize([c[2] for c in C]),
                     "rss_after_load_mb": round(statistics.mean([c[3] for c in C]), 1)}
        out["models"][mid] = m
        log(f"{tag} warm_load_med={m['warm']['load_s']['median']}s "
            f"cold_load_med={m['cold']['load_s']['median']}s "
            f"rss~{m['cold']['rss_after_load_mb']}MB")
        del tok
        gc.collect()

    # Memory residency: brain alone, brain+1 skill, brain+2 skills, +KV growth
    log("memory residency test start")
    mem = {"baseline_rss_mb": round(rss_mb(), 1)}
    tok = AutoTokenizer.from_pretrained(MODELS[2][0])
    brain = AutoModelForCausalLM.from_pretrained(MODELS[2][0])
    brain.eval()
    mem["brain_rss_mb"] = round(rss_mb(), 1)
    ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        brain.generate(ids, max_new_tokens=64, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    mem["brain_plus_kv64_rss_mb"] = round(rss_mb(), 1)
    s1 = AutoModelForCausalLM.from_pretrained(MODELS[0][0]); s1.eval()
    mem["brain_skill1_rss_mb"] = round(rss_mb(), 1)
    s2 = AutoModelForCausalLM.from_pretrained(MODELS[1][0]); s2.eval()
    mem["brain_skill1_skill2_rss_mb"] = round(rss_mb(), 1)
    del s1, s2, brain, tok
    gc.collect()
    mem["after_free_rss_mb"] = round(rss_mb(), 1)
    out["memory"] = mem
    log(f"memory: {mem}")

    with open("bench_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log("ALL DONE")

if __name__ == "__main__":
    main()
