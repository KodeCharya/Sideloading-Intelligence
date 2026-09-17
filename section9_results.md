# Section 9: Preliminary Empirical Results (prototype appendix)

*Status: draft insert for the "Sideloading Intelligence" paper. All figures below
are measured on the hardware and software described in Section 9.4. They replace
nothing in the main text by themselves; we indicate explicitly where they support,
complicate, or contradict the hypothetical illustrations in Table 1, Table 2, and
Sections 5.1–5.2. The prototype is minimal by design, and Section 9.5 states what
it does not establish.*

## 9.1 Swap latency: measured load-to-first-token times

We measured wall-clock time to load a checkpoint from disk into RAM
(`AutoModelForCausalLM.from_pretrained`, CPU) plus time to produce the first
token (greedy generation, one token, fixed diagnostic prompt), for three
checkpoints spanning the illustrative size range of the paper: 353 MB, 1520 MB,
and 2200 MB on disk. Each condition was repeated 12 times; we report
mean / median / stdev throughout. "Warm" means the file was resident in the OS
page cache (repeated back-to-back loads). "Cold" means a 12 GB sequential cache
turnover was performed before every repetition, verified to return the
subsequent read to disk speed (Section 9.4). An earlier attempt using an 8 GB
in-process touch buffer as the eviction mechanism is worth reporting as a
negative methodological result: it produced cold timings indistinguishable from
warm (ratio ~1.0), presumably because allocation without I/O pressure does not
displace standby cache pages on this OS. Those runs were discarded.

**Table 4. Measured swap latency (seconds), 12 repetitions per cell.**

| Checkpoint (params, dtype, file size) | Cold load mean / med / sd | Warm load mean / med / sd | Cold total¹ mean / med | Warm total¹ mean / med |
|---|---|---|---|---|
| distilgpt2 (82M, fp32, 353 MB) | 1.55 / 1.50 / 0.19 | 0.71 / 0.71 / 0.01 | 1.59 / 1.53 | 0.73 / 0.73 |
| gpt2-medium (355M, fp32, 1520 MB) | 4.72 / 4.70 / 0.08 | 1.51 / 1.49 / 0.04 | 4.85 / 4.84 | 1.62 / 1.61 |
| TinyLlama-1.1B-Chat (1.1B, fp16, 2200 MB) | 6.88 / 6.89 / 0.13 | 1.99 / 1.97 / 0.07 | 8.27 / 8.31 | 3.13 / 3.10 |

¹ "Total" = checkpoint load plus first-token generation (median first-token
compute alone: 0.03 s / 0.14 s / 1.39 s cold for the three sizes; load dominates
I/O-bound cold cost, while compute dominates nothing — first-token compute is a
small fraction in all cells).

Three observations. First, cold loads are 2–3.5× slower than warm loads at every
size, and scale roughly with file size (≈ 3–4 s/GB cold on this SATA SSD).
Second, absolute cold latencies are large relative to interactive expectations:
a single brain-scale checkpoint costs ~7 s cold. Third, the Section 4.2 scenario
(load four specialists on a multi-domain query) has a measured analogue: loading
our three checkpoints cold in sequence costs
1.53 + 4.84 + 8.31 ≈ **14.7 s** (median-of-medians sum), and ≈ 5.4 s warm.
Extrapolating to four skills on the same hardware gives roughly 15–20 s cold.
We state this plainly as a complication of Section 5.1: the paper frames swap
latency as a "measurable" cost to be softened by a hot set and predictive
preloading. The measurements suggest the hot set is not an optimization but a
necessity — any cold fan-out over two or more skills exceeds plausible
per-turn latency budgets — and that predictive preloading must hide on the
order of 5–8 s per skill, a substantially harder bar than the text implies.
Conversely, and in the paper's favour, none of our checkpoints are quantized;
Q4/Q8 weights would be roughly 2–4× smaller on disk, so Table 4 should be read
as a conservative upper bound for quantized equivalents, not as a direct
refutation. We did not test quantized formats, and the extent to which
quantization closes the gap is unmeasured here.

## 9.2 Routing accuracy: a 60-query test of the embedding-similarity router

We implemented the simplest router described in Section 3.2: cosine similarity
between the query and each skill-registry description, using TF-IDF vectors
(fit on registry descriptions plus fixed background paragraphs only; test
queries unseen at fit time). This is deliberately a weak, sparse instantiation
of "embedding similarity," and should be read as a lower bound on what a dense
embedding router might achieve. The test set contains 60 hand-written queries:
40 clear single-domain queries (10 each for linux, networking, python,
cybersecurity), 10 ambiguous cross-domain queries with a best-single-skill gold
label (in the style of the Section 4.2 scenario), and 10 off-topic queries for
which the correct behaviour is fallback (no skill routed).

**Table 5. Routing accuracy of the TF-IDF similarity router (n = 60).**

| Subset | Top-1 | Top-2 |
|---|---|---|
| Clear single-domain (n = 40) | 0.60 (24/40) | 0.75 (30/40) |
| Ambiguous cross-domain (n = 10) | 0.40 (4/10) | 0.60 (6/10) |
| Per-domain top-1 (10 each) | linux 9/10; networking 5/10; python 4/10; cybersecurity 6/10 | — |

Notably, top-2 recall repairs nearly all networking errors (10/10) and the
remaining linux error, while python errors persist at top-2 (4/10) — vocabulary
overlap between the python registry description and general programming queries
is the weakest link in this registry, an artifact of description wording as
much as of the method.

Mean top-1 similarity was 0.141 for correctly routed answerable queries, 0.047
for incorrectly routed ones, and 0.038 for off-topic queries: correct routes do
score higher on average, but the distributions overlap substantially (incorrect
and off-topic means are nearly identical), so no threshold cleanly separates
answerable from unanswerable queries. At threshold 0.0 (route everything), all
10 off-topic queries are misrouted, a 0.53 wrong-invocation rate overall.

For the fallback claim of Section 5.2 we swept the loading threshold, routing
only when the top score exceeds it and otherwise asking a clarifying question,
with "annoyance" approximated as the fallback rate on answerable queries:

**Table 6. Fallback-threshold sweep (60 queries; 50 answerable, 10 off-topic).**

| Threshold | Routed | Wrong-invocation rate | Annoyance (fallback on answerable) | Off-topic suppressed |
|---|---|---|---|---|
| 0.00 | 60 | 0.53 | 0.00 | 0/10 |
| 0.05 | 32 | 0.28 | 0.40 | 8/10 |
| 0.10 | 28 | 0.29 | 0.48 | 8/10 |
| 0.15 | 16 | 0.25 | 0.72 | 8/10 |
| 0.25 | 4 | 0.00 | 0.92 | 10/10 |

Fallback does reduce wrong-skill invocations — roughly halving them at
threshold 0.05 — but at a steep price: 40% of answerable queries trigger a
clarifying question, rising to near-total suppression of the system at higher
thresholds. We regard this as corroborating the *existence* of the trade-off
posited in Section 5.2 while complicating any assumption that it is gentle, at
least for a weak sparse router: the operating point that halves errors annoys
nearly half the time. Whether dense embeddings or a trained classification head
flatten this curve is the obvious, and entirely open, follow-up; nothing here
licenses the conclusion that fallback is impractical in general.

## 9.3 Memory budget: measured residency versus Table 1

Process resident set size (RSS, measured via `psutil`) was recorded at each
stage of a cumulative residency test: framework baseline (torch + transformers
imported, no model), brain resident, brain after 64 generated tokens (KV cache
effect), brain plus one small skill, and brain plus two skills.

**Table 7. Measured RSS (MB) against the hypothetical Table 1 budget.**

| Stage | Measured RSS | Table 1 analogue |
|---|---|---|
| Framework baseline (no model) | 359 | (part of "runtime overhead") |
| + Brain (TinyLlama-1.1B, 2200 MB weights) | 2571 (+2212) | Brain 2.5 GB |
| + 64 generated tokens (KV cache) | 2573 (+2) | (part of "KV cache and overhead" 500 MB) |
| + Skill 1 (distilgpt2, 353 MB weights) | 2901 (+330) | Skill 300 MB |
| + Skill 1 + Skill 2 (gpt2-medium, 1520 MB) | 4319 (+1418) | Skills 300 MB each |
| After freeing all models | 362 (≈ baseline) | — |

The shape of Table 1 survives contact with measurement remarkably well: the
brain-scale checkpoint occupies 2.57 GB against the illustrated 2.5 GB, and each
skill resides at roughly 0.93–1.00× its file size, i.e. a 300 MB skill would
occupy on the order of 300 MB as the paper assumes. Two qualifications are
required, both of which revise Section 7-worthy detail rather than the
architecture. First, the ~360 MB framework baseline (interpreter, torch,
transformers) is a fixed cost the paper's budget does not name; on a 4 GB-class
device budget it is material. Second, the paper's combined 500 MB line for "KV
cache and runtime overhead" looks generous at this scale and sequence length:
64 generated tokens grew RSS by ~2 MB for the 1.1B model. KV growth is linear
in sequence length and would matter for long contexts or batched serving, which
we did not test; the claim here is only that short single-turn exchanges do not
reproduce a 500 MB overhead on this stack. Our measured total (brain + two
skills, one of them much larger than the paper's illustrative 300 MB skill)
is 4.3 GB against the paper's 3.9 GB for brain + three small skills — same
order of magnitude, reached by a different composition, and therefore weakly
supportive rather than confirmatory.

## 9.4 Replicability: hardware, models, and software

Hardware: Dell OptiPlex 3050, Intel i7-7700 (4C/8T, 3.6 GHz), 16 GB DDR4
(2×8 GB, mixed 2667/2133 MT/s), Intel HD 630 + NVIDIA GT 710 (neither used;
all inference CPU-only), OS drive and model cache on Kingston SA400S3 120 GB
SATA SSD (drive C:), Windows 10 Enterprise LTSC build 19044. Available RAM at
test time ≈ 10.8 GB. Software: Python 3.14.7, torch 2.14.0+cpu (4 intra-op
threads, library default on this machine), transformers 5.17.0, safetensors
0.8.0, tokenizers 0.23.2, accelerate 1.15.0, scikit-learn 1.9.1, numpy 2.5.3,
psutil 7.2.2, huggingface_hub 1.31.0. Models (unmodified Hub checkpoints, exact
revisions as cached June–Sept 2026): `distilgpt2` (82M params, fp32,
model.safetensors 352.8 MB), `gpt2-medium` (355M, fp32, 1520.0 MB),
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B, fp16, 2200.1 MB). No quantization
was applied; dtypes are as stored upstream. Fixed prompt for generation timing:
"Why is my Linux server consuming 100% CPU?" (prefill + 1 greedy token for
Table 4; 64 tokens for the KV row of Table 7). Cold condition: before each
repetition, two 6 GB eviction files in `%TEMP%` were read sequentially
(12 GB turnover, ~30 s), which returned a probe read of the 2.2 GB weights
file from ~1 s (cached) to ~7 s (disk speed); per-repetition timings, scripts
(`bench_all.py`, `bench_cold.py`, `router_test.py`, `evict_probe.py`) and raw
outputs (`bench_results.json`, `bench_cold.json`, `bench_reps.jsonl`,
`router_results.json`) accompany this section.

## 9.5 What this prototype does not establish

This prototype replaces the paper's hypothetical numbers with measured ones at
exactly one point in the design space, and it should not be read as validating
the architecture. The routing test covers five domains with 60 queries; it says
nothing about the "tens or hundreds of specialists" regime of Section 3.6, for
which a flat similarity router would face a harder discrimination problem than
anything measured here. The router is TF-IDF, not dense embeddings or a trained
head, so Table 5 bounds the weakest viable router, not the router the paper
would presumably deploy. The checkpoints are unquantized fp32/fp16 models, not
the quantized skills Table 1 assumes, and none is a genuine domain specialist —
sizes proxy for skills, not competence. Latency and memory were measured for
sequential loads on one SATA-SSD CPU-only machine; NVMe storage, quantized
formats, concurrent loading, a real hot-set/eviction policy, multi-turn
dialogue, the verification layer (3.7), adapter personalization (3.5), and any
user-facing evaluation of fallback annoyance (5.2) or arbitration (5.3) are all
untested. The sample sizes (12 repetitions, 60 queries) support the qualitative
conclusions drawn above — cold fan-out is slow, the fallback curve is steep for
weak routers, residency tracks file size — and nothing stronger. In particular,
the single most load-bearing unknown identified in Section 7 remains unknown:
whether a few-hundred-million-parameter skill is actually reliable inside its
declared domain. No result here bears on that question.
