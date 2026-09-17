"""One-time setup + validation for true cold-cache reads.
Creates 2x6GB eviction files, then proves cache turnover works by timing a raw
sequential read of the 2.2GB TinyLlama weights hot vs after turnover.
"""
import os, time

os.environ["HF_HUB_DISABLE_XET"] = "1"
TMP = os.environ.get("TEMP", r"C:\Users\MUKESH\AppData\Local\Temp")
EV = [os.path.join(TMP, "evict1.bin"), os.path.join(TMP, "evict2.bin")]

def raw_read(path, chunk=4 * 2**20):
    t0 = time.perf_counter()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            n += len(b)
    return time.perf_counter() - t0, n

def turnover():
    for p in EV:
        raw_read(p)

if __name__ == "__main__":
    for p in EV:
        if not (os.path.exists(p) and os.path.getsize(p) == 6 * 2**30):
            print(f"creating {p} ...", flush=True)
            with open(p, "wb") as f:
                for _ in range(6 * 2**10 // 4):  # 6GB in 4MB urandom chunks
                    f.write(os.urandom(4 * 2**20))
            print("created", flush=True)
    from huggingface_hub import snapshot_download
    snap = snapshot_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                             allow_patterns=["*.safetensors"])
    w = os.path.join(snap, "model.safetensors")
    t1, n1 = raw_read(w)   # hot (just downloaded/used)
    print(f"hot read: {t1:.2f}s for {n1/1e9:.2f}GB = {n1/1e9/t1:.0f} MB/s", flush=True)
    t0 = time.perf_counter()
    turnover()
    print(f"turnover 12GB took {time.perf_counter()-t0:.1f}s", flush=True)
    t2, n2 = raw_read(w)
    print(f"post-turnover read: {t2:.2f}s for {n2/1e9:.2f}GB = {n2/1e9/t2:.0f} MB/s", flush=True)
    print("COLD/WARM ratio:", round(t2 / max(t1, 1e-6), 2), flush=True)
