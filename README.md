# Sideloading Intelligence

**A hierarchical Brain–Skills architecture for memory-constrained local language model systems.**

Sideloading Intelligence is a conceptual architecture (plus a minimal empirical prototype) for running AI systems locally under tight memory budgets. Instead of keeping one large general-purpose model resident at all times, a small planner model (the **Brain**, ~1–4B parameters) routes each request to one or more compact domain specialists (**Skills**, hundreds of millions of parameters) that are loaded into memory on demand and evicted when no longer needed — an OS-inspired residency policy applied to whole models rather than to a single model's KV cache.

This repo contains the paper, the preliminary benchmark prototype, and the raw results referenced in it.

> **Status:** This is primarily a design paper and research agenda, not a validated system. A minimal single-machine prototype (see below) provides first measurements for two of the five open problems the paper identifies; most of the architecture remains unimplemented. See Section 7.5 of the paper for an explicit account of what the prototype does and does not establish.

---

## What's in this repo

| File | Description |
|---|---|
| `sideloading-intelligence-paper.docx` | Full paper: architecture, five open research problems, and Section 7 preliminary empirical results. |
| `bench_all.py` | Swap-latency + memory benchmark (first-pass method; cold-cache eviction via in-process buffer touch — see note below). |
| `bench_cold.py` | Corrected swap-latency benchmark using verified 12 GB cache-turnover eviction. |
| `evict_probe.py` | One-time setup/validation script proving the cache-eviction method actually displaces the OS page cache. |
| `router_test.py` | Routing-accuracy and fallback-threshold test using a TF–IDF similarity router over 60 hand-labeled queries. |
| `dl_models.py` | Downloads the three benchmark checkpoints used throughout. |
| `bench.log`, `bench_cold.log`, `dl.log` | Raw run logs. |
| `bench_results.json`, `bench_cold.json`, `bench_reps.jsonl`, `router_results.json` | Aggregated and per-repetition results. |
| `section9_results.md` | Draft write-up of the empirical results (source material for the paper's Section 7). |

## Key preliminary results

- **Swap latency:** cold-loading a 1.1B-parameter checkpoint takes ~6.9s on a SATA SSD; loading three checkpoints of increasing size in sequence costs ~14.7s cold vs. ~5.4s warm — suggesting a "hot set" of never-evicted skills is closer to a necessity than an optimization.
- **Routing accuracy:** a simple TF–IDF similarity router hits 60% top-1 / 75% top-2 accuracy on clear single-domain queries (40 queries, 4 domains); fallback thresholding roughly halves wrong-skill invocations but at the cost of declining ~40% of answerable queries.
- **Memory residency:** measured process memory tracks the paper's hypothetical Table 1 budget reasonably closely (Brain 2.57 GB measured vs. 2.5 GB illustrative).

None of this speaks to the architecture's central open question: whether a compact model can be relied on as a genuine domain specialist. See the paper for full caveats.

## Reproducing the benchmarks

```bash
pip install torch transformers accelerate scikit-learn psutil numpy huggingface_hub

python dl_models.py       # downloads distilgpt2, gpt2-medium, TinyLlama-1.1B-Chat
python evict_probe.py     # one-time: creates eviction files, validates cold-cache method
python bench_cold.py      # swap-latency benchmark (12 reps/checkpoint)
python router_test.py     # routing-accuracy + fallback-threshold sweep
```

Hardware used for reported numbers: Intel i7-7700 (4C/8T), 16 GB DDR4, SATA SSD, CPU-only inference. Your numbers will vary with storage type, CPU, and available RAM — that's expected and part of the point.

## Open problems (see paper Section 5)

1. Swap latency and scheduling
2. Routing uncertainty and fallback behavior
3. Cross-specialist arbitration
4. Trust and provenance in a skill "marketplace"
5. Evaluation methodology for compact specialists

## Contributing

Issues and PRs welcome, especially around:
- Replacing the TF–IDF router with a dense embedding or trained classifier
- Testing quantized checkpoints
- Building genuine narrow-domain specialists (not generic small models) to test in-domain competence
- Implementing an actual eviction/hot-set policy under multi-turn workloads

## Citation

```bibtex
@misc{sideloading-intelligence,
  author = {KodeCharya},
  title  = {Sideloading Intelligence: A Hierarchical Brain–Skills Architecture for Memory-Constrained Local Language Model Systems},
  year   = {2026}
}
```
