"""Routing accuracy test for the Sideloading Intelligence prototype (paper Sec 3.2 / 5.2).

Router: TF-IDF cosine similarity between query and skill-registry descriptions
(sparse-embedding instantiation of the paper's 'embedding similarity' router).
Vectorizer is fit ONLY on registry descriptions + fixed background paragraphs
(test queries are never seen at fit time).
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS = {
    "linux": (
        "Linux system administration skill. Handles shell commands, systemd services, "
        "process management, CPU and memory diagnostics, log files, file permissions, "
        "package management, cron jobs, kernel modules, and server troubleshooting."
    ),
    "networking": (
        "Computer networking skill. Handles TCP IP DNS DHCP routing, nginx and web server "
        "configuration, latency and packet loss diagnosis, firewalls, ports, load balancers, "
        "bandwidth monitoring, and connectivity troubleshooting."
    ),
    "python": (
        "Python programming skill. Handles Python syntax, debugging tracebacks, pip packages, "
        "virtual environments, data structures, NumPy pandas scripting, writing functions, "
        "and fixing Python errors."
    ),
    "cybersecurity": (
        "Cybersecurity skill. Handles intrusion detection, malware analysis, suspicious logins, "
        "DDoS and network attacks, vulnerability assessment, access anomalies, encryption, "
        "and incident response."
    ),
    "general": (
        "General knowledge skill. Handles everyday questions, cooking, travel, history, "
        "sports, weather, and topics outside any specialist domain."
    ),
}

# Fixed background corpus (fit-time only, disjoint from test queries).
BACKGROUND = [
    "Use systemctl to restart a failed systemd service and journalctl to inspect its logs.",
    "Check CPU usage per process with top or htop and review load averages over time.",
    "DNS resolves hostnames to IP addresses; test with dig or nslookup commands.",
    "A reverse proxy like nginx forwards client requests to backend application servers.",
    "Python tracebacks show the call stack; read the last line for the exception type.",
    "Create a virtual environment with python minus m venv and install packages via pip.",
    "Brute force login attempts appear as repeated authentication failures in auth logs.",
    "A SYN flood overwhelms a server with half open TCP connections, a form of DDoS.",
    "How to bake bread at home with simple ingredients and easy steps.",
    "Plan a weekend trip itinerary with sightseeing and restaurant recommendations.",
]

# (query, gold_label); gold None = off-topic, correct action is fallback.
QUERIES = [
    # linux (10)
    ("Why is my Linux server consuming 100% CPU?", "linux"),
    ("How do I restart a failed systemd service on Ubuntu?", "linux"),
    ("Which command shows disk usage per directory?", "linux"),
    ("My cron job runs but produces no output, how do I debug it?", "linux"),
    ("How to change file permissions recursively with chmod?", "linux"),
    ("Where are kernel boot messages stored after startup?", "linux"),
    ("How do I list all running processes sorted by memory?", "linux"),
    ("apt update fails with a GPG error, what should I check?", "linux"),
    ("How to mount an external drive at boot with fstab?", "linux"),
    ("What does load average 8.0 mean on a 4-core machine?", "linux"),
    # networking (10)
    ("Is this latency spike caused by packet loss or DNS?", "networking"),
    ("How do I configure nginx as a reverse proxy?", "networking"),
    ("What does a TCP RST packet indicate during a connection?", "networking"),
    ("My server is unreachable by hostname but reachable by IP.", "networking"),
    ("How to open port 443 in the firewall for HTTPS?", "networking"),
    ("Diagnose high bandwidth usage on a network interface.", "networking"),
    ("What is DHCP and why did my machine get a new address?", "networking"),
    ("How to test whether a remote port is open and listening?", "networking"),
    ("Why do traceroute hops time out in the middle of the path?", "networking"),
    ("How to set up a static route for a second subnet?", "networking"),
    # python (10)
    ("How do I fix a ValueError when parsing dates in pandas?", "python"),
    ("What does the traceback 'list index out of range' mean?", "python"),
    ("How to create a virtual environment and install requirements?", "python"),
    ("Why is my dict lookup slow inside a large loop?", "python"),
    ("How do I read a CSV file with NumPy?", "python"),
    ("pip install fails with SSL certificate errors, how to fix?", "python"),
    ("How to catch multiple exception types in one except block?", "python"),
    ("What is the difference between a list and a tuple?", "python"),
    ("How do I plot two lines with matplotlib and a legend?", "python"),
    ("Why does my recursive function hit RecursionError?", "python"),
    # cybersecurity (10)
    ("Are repeated SSH login failures a brute-force attack?", "cybersecurity"),
    ("How to tell if high CPU usage is caused by malware?", "cybersecurity"),
    ("What are indicators of a DDoS attack in nginx access logs?", "cybersecurity"),
    ("A user reports a suspicious login from another country.", "cybersecurity"),
    ("How to check a binary for known malware signatures?", "cybersecurity"),
    ("What steps follow a suspected credential leak?", "cybersecurity"),
    ("How to detect ARP spoofing on a local network?", "cybersecurity"),
    ("Is an unexpected outbound connection a sign of compromise?", "cybersecurity"),
    ("How to audit sudo usage for privilege escalation attempts?", "cybersecurity"),
    ("What is the incident response process for ransomware?", "cybersecurity"),
    # ambiguous / cross-domain but answerable (10) — gold = best single skill
    ("My Debian server has high CPU, nginx is running, is it a network attack?", "cybersecurity"),
    ("A Python script saturates one CPU core on the server.", "python"),
    ("SSH logins are slow; is it DNS or an attack?", "cybersecurity"),
    ("nginx returns 502 after I deployed a Python app.", "networking"),
    ("A cron job running a backup script fills the disk overnight.", "linux"),
    ("pip download speeds are very slow on this machine.", "networking"),
    ("Suspicious process listens on an open port; how to investigate?", "linux"),
    ("Python requests to an HTTPS endpoint fail with certificate errors.", "python"),
    ("Firewall blocks my Python package mirror; what rule to add?", "networking"),
    ("Audit logs show odd sudo calls from a service account.", "cybersecurity"),
    # off-topic (10) — correct action is fallback (no skill above threshold)
    ("What is a good recipe for sourdough bread?", None),
    ("Who won the football match last night?", None),
    ("Suggest a three-day itinerary for Rome.", None),
    ("How do I knit a sweater for beginners?", None),
    ("What is the capital of Assyria in ancient history?", None),
    ("Tips for photographing the northern lights?", None),
    ("How to care for a new houseplant?", None),
    ("Recommend a science fiction novel about space travel.", None),
    ("What are the rules of chess castling?", None),
    ("How to plan a wedding on a small budget?", None),
]

def main():
    names = list(SKILLS)
    descs = [SKILLS[n] for n in names]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vec.fit(descs + BACKGROUND)
    D = vec.transform(descs)

    rows = []
    for q, gold in QUERIES:
        s = cosine_similarity(vec.transform([q]), D)[0]
        order = np.argsort(-s)
        rows.append({
            "query": q, "gold": gold,
            "top1": names[int(order[0])], "top1_score": float(s[order[0]]),
            "top2": names[int(order[1])], "top2_score": float(s[order[1]]),
            "scores": {n: float(v) for n, v in zip(names, s)},
        })
    for r in rows:
        r["top1_correct"] = (r["gold"] is not None and r["top1"] == r["gold"])
        r["top2_correct"] = (r["gold"] is not None and r["gold"] in (r["top1"], r["top2"]))

    clear = rows[0:40]
    amb = rows[40:50]
    off = rows[50:60]
    assert len(clear) == 40 and len(amb) == 10 and len(off) == 10, (len(clear), len(amb), len(off))

    def stats(rs, key):
        v = np.array([r[key] for r in rs])
        return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                    stdev=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    min=float(v.min()), max=float(v.max()))

    top1_acc = sum(r["top1_correct"] for r in clear) / len(clear)
    top2_acc = sum(r["top2_correct"] for r in clear) / len(clear)
    amb_top1 = sum(r["top1_correct"] for r in amb) / len(amb)
    amb_top2 = sum(r["top2_correct"] for r in amb) / len(amb)

    # Fallback sweep: route iff top1_score >= thr, else ask clarifying question.
    # Wrong invocation = routed to a wrong skill (incl. routing off-topic anywhere).
    # Annoyance proxy = fallback triggered on an answerable query (clear+amb).
    answerable = clear + amb
    sweep = []
    for thr in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        routed = [r for r in rows if r["top1_score"] >= thr]
        wrong = [r for r in routed if r["gold"] is None or r["top1"] != r["gold"]]
        fb_ans = [r for r in answerable if r["top1_score"] < thr]
        fb_off = [r for r in off if r["top1_score"] < thr]  # correctly suppressed
        sweep.append(dict(
            threshold=thr,
            routed=len(routed),
            wrong_invocations=len(wrong),
            wrong_rate=round(len(wrong) / len(routed), 3) if routed else 0.0,
            annoyance_rate=round(len(fb_ans) / len(answerable), 3),
            offtopic_suppressed=f"{len(fb_off)}/10",
        ))

    correct_scores = [r["top1_score"] for r in clear + amb if r["top1_correct"]]
    incorrect_scores = [r["top1_score"] for r in clear + amb if not r["top1_correct"]]
    off_scores = [r["top1_score"] for r in off]
    per_domain = {}
    for d in ("linux", "networking", "python", "cybersecurity"):
        rs = [r for r in clear if r["gold"] == d]
        per_domain[d] = dict(top1=f"{sum(r['top1_correct'] for r in rs)}/{len(rs)}",
                             top2=f"{sum(r['top2_correct'] for r in rs)}/{len(rs)}")

    out = dict(
        n_queries=len(rows), n_clear=len(clear), n_ambiguous=len(amb), n_offtopic=len(off),
        top1_accuracy_clear=round(top1_acc, 3),
        top2_accuracy_clear=round(top2_acc, 3),
        top1_accuracy_ambiguous=round(amb_top1, 3),
        top2_accuracy_ambiguous=round(amb_top2, 3),
        per_domain=per_domain,
        top1_score_correct=stats(clear + amb and [r for r in clear + amb if r["top1_correct"]], "top1_score"),
        top1_score_incorrect=stats([r for r in clear + amb if not r["top1_correct"]], "top1_score"),
        top1_score_offtopic=stats(off, "top1_score"),
        fallback_sweep=sweep,
        rows=rows,
    )
    with open("router_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"n={len(rows)} clear_top1={top1_acc:.3f} clear_top2={top2_acc:.3f} "
          f"amb_top1={amb_top1:.3f} amb_top2={amb_top2:.3f}")
    print("per-domain(top1):", {k: v["top1"] for k, v in per_domain.items()})
    print("mean top1 score correct=%.3f incorrect=%.3f offtopic=%.3f" % (
        np.mean(correct_scores), np.mean(incorrect_scores), np.mean(off_scores)))
    print("sweep (thr, wrong_rate, annoyance, offtopic_suppressed):")
    for s in sweep:
        print(f"  {s['threshold']:.2f} routed={s['routed']:3d} wrong_rate={s['wrong_rate']:.3f} "
              f"annoyance={s['annoyance_rate']:.3f} offsup={s['offtopic_suppressed']}")

if __name__ == "__main__":
    main()
