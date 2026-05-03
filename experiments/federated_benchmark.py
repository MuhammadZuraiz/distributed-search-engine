"""
Federated Search Benchmark
============================
Measures query latency as number of shards varies.
Demonstrates that more shards = faster per-shard search
(each shard searches fewer documents).
"""

import sys
import os
import json
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.federated import shard_index, ShardSearcher, FederatedSearchCoordinator

INDEX_PATH = "data/index/inverted_index.json"
PR_PATH    = "data/index/pagerank.json"

print("Loading index...")
with open(INDEX_PATH, encoding="utf-8") as f:
    full_index = json.load(f)

pr = {}
if os.path.exists(PR_PATH):
    with open(PR_PATH, encoding="utf-8") as f:
        pr = {int(k): v for k, v in json.load(f).items()}

TEST_QUERIES = [
    ["distributed", "computing"],
    ["parallel", "systems"],
    ["web", "crawler"],
    ["python", "programming"],
    ["machine", "learning"],
    ["network", "protocol"],
    ["database", "storage"],
    ["algorithm", "performance"],
]

shard_counts   = [1, 2, 3, 6]
latencies      = []
terms_per_shard = []

for n_shards in shard_counts:
    shards    = shard_index(full_index, n_shards)
    searchers = [ShardSearcher(s, i, pr) for i, s in enumerate(shards)]

    terms_per_shard.append(
        sum(len(s) for s in shards) / n_shards
    )

    times = []
    for query_terms in TEST_QUERIES:
        t0 = time.perf_counter()

        # simulate parallel shard search (sequential here, parallel in MPI)
        all_results = []
        for searcher in searchers:
            results = searcher.search(query_terms, top_k=10)
            all_results.append(results)

        # merge
        flat = [(s, d) for res in all_results for s, d in res]
        flat.sort(reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    avg_ms = sum(times) / len(times)
    latencies.append(avg_ms)
    print(f"  {n_shards} shards: {avg_ms:.2f}ms avg, "
          f"{terms_per_shard[-1]:.0f} terms/shard")

# ── plot ──────────────────────────────────────────────────────────────────────

BLUE  = "#2E75B6"
TEAL  = "#1D9E75"
GRID  = "#E8ECF0"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID,
    "figure.dpi": 150
})

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Federated Search — Query Latency vs Shard Count",
             fontsize=13, fontweight="bold", color="#1F4E79")

ax = axes[0]
ax.plot(shard_counts, latencies, "o-", color=BLUE, linewidth=2.5,
        markersize=8)
for x, y in zip(shard_counts, latencies):
    ax.annotate(f"{y:.1f}ms", (x, y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=9, color="#0C447C", fontweight="bold")
ax.set_xlabel("Number of shards")
ax.set_ylabel("Avg query latency (ms)")
ax.set_title("Query latency vs shards", fontweight="bold", color="#1F4E79")
ax.set_xticks(shard_counts)

ax = axes[1]
ax.bar(shard_counts, [t/1000 for t in terms_per_shard],
       color=TEAL, width=0.6, zorder=3, edgecolor="white")
for x, y in zip(shard_counts, terms_per_shard):
    ax.text(x, y/1000 + 0.5, f"{y/1000:.1f}K",
            ha="center", fontsize=9,
            fontweight="bold", color="#085041")
ax.set_xlabel("Number of shards")
ax.set_ylabel("Terms per shard (thousands)")
ax.set_title("Index size per shard\n(smaller = faster per-shard search)",
             fontweight="bold", color="#1F4E79")
ax.set_xticks(shard_counts)

plt.tight_layout()
out = "experiments/graphs/federated_search_benchmark.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nGraph saved: {out}")
plt.show()