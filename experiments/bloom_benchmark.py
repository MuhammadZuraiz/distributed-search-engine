"""
Bloom Filter Benchmark
========================
Measures and plots:
  1. Memory usage vs capacity
  2. False positive rate vs items inserted
  3. Empirical FP rate vs theoretical FP rate
  4. Memory: Python set vs Bloom filter comparison

Run:
    python experiments/bloom_benchmark.py
"""

import sys
import os
import time
import random
import string
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.bloom_filter import BloomFilter

os.makedirs("experiments/graphs", exist_ok=True)


def random_url(length=40):
    chars = string.ascii_lowercase + string.digits
    path  = "".join(random.choices(chars, k=length))
    return f"https://en.wikipedia.org/wiki/{path}"


# ── experiment 1: memory vs capacity ─────────────────────────────────────────

capacities   = [10_000, 50_000, 100_000, 500_000, 1_000_000]
bloom_memory = []
set_memory   = []

for cap in capacities:
    bf = BloomFilter(cap, false_positive_rate=0.001)
    bloom_memory.append(bf.memory_bytes() / 1024 / 1024)

    # Python set memory: ~200 bytes per average URL string
    avg_url_len = 50
    set_memory.append(cap * (avg_url_len + 50) / 1024 / 1024)

# ── experiment 2: empirical false positive rate ───────────────────────────────

bf         = BloomFilter(capacity=10_000, false_positive_rate=0.01)
insert_n   = 10_000
test_n     = 5_000

# insert known URLs
known = [random_url() for _ in range(insert_n)]
for url in known:
    bf.add(url)

# test with URLs we know were NOT inserted
unseen      = [random_url(length=45) for _ in range(test_n)]
fp_count    = sum(1 for url in unseen if url in bf)
empirical_fp = fp_count / test_n

print(f"\nEmpirical FP rate : {empirical_fp:.4%}")
print(f"Theoretical FP rate: {bf.current_fp_rate():.4%}")
print(f"Memory used: {bf.memory_bytes()/1024:.1f} KB")
print(f"Python set equiv: {insert_n * 250 / 1024:.1f} KB")

# ── experiment 3: FP rate as items inserted grows ─────────────────────────────

bf2         = BloomFilter(capacity=50_000, false_positive_rate=0.01)
checkpoints = list(range(0, 50_001, 2500))
fp_rates    = []
load_factors = []

all_urls  = [random_url() for _ in range(50_000)]
test_urls = [random_url(length=46) for _ in range(1000)]

inserted = 0
for cp in checkpoints:
    while inserted < cp:
        bf2.add(all_urls[inserted])
        inserted += 1
    fp = sum(1 for u in test_urls if u in bf2) / len(test_urls)
    fp_rates.append(fp * 100)
    load_factors.append(bf2.load_factor() * 100)

# ── experiment 4: insert + lookup speed ──────────────────────────────────────

bf3      = BloomFilter(capacity=100_000, false_positive_rate=0.001)
urls_spd = [random_url() for _ in range(10_000)]

t0 = time.time()
for u in urls_spd:
    bf3.add(u)
insert_time = (time.time() - t0) / len(urls_spd) * 1e6

t0 = time.time()
for u in urls_spd:
    _ = u in bf3
lookup_time = (time.time() - t0) / len(urls_spd) * 1e6

print(f"Insert speed: {insert_time:.2f} µs/URL")
print(f"Lookup speed: {lookup_time:.2f} µs/URL")

# ── plot ──────────────────────────────────────────────────────────────────────

BLUE  = "#2E75B6"
TEAL  = "#1D9E75"
AMBER = "#BA7517"
CORAL = "#D85A30"
GRID  = "#E8ECF0"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.dpi": 150
})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "Bloom Filter — Memory vs Accuracy Trade-off Analysis",
    fontsize=14, fontweight="bold", color="#1F4E79", y=1.01
)

# plot 1: memory comparison
ax = axes[0, 0]
x  = np.arange(len(capacities))
w  = 0.35
ax.bar(x - w/2, bloom_memory, w, label="Bloom filter", color=TEAL,
       zorder=3, edgecolor="white")
ax.bar(x + w/2, set_memory,   w, label="Python set()",  color=CORAL,
       zorder=3, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"{c//1000}K" for c in capacities])
ax.set_xlabel("Capacity (URLs)")
ax.set_ylabel("Memory (MB)")
ax.set_title("Memory: Bloom filter vs Python set",
             fontweight="bold", color="#1F4E79")
ax.legend(fontsize=9)
for i, (bm, sm) in enumerate(zip(bloom_memory, set_memory)):
    ax.text(i - w/2, bm + 0.3, f"{bm:.1f}MB", ha="center",
            fontsize=8, color="#085041", fontweight="bold")

# plot 2: FP rate as load increases
ax = axes[0, 1]
ax.plot(checkpoints, fp_rates, "o-", color=CORAL, linewidth=2,
        markersize=4, label="Empirical FP rate")
ax2 = ax.twinx()
ax2.plot(checkpoints, load_factors, "--", color=BLUE, linewidth=1.5,
         label="Bit array load factor")
ax2.set_ylabel("Load factor (%)", color=BLUE)
ax2.tick_params(axis="y", labelcolor=BLUE)
ax2.spines["top"].set_visible(False)
ax.set_xlabel("URLs inserted")
ax.set_ylabel("False positive rate (%)", color=CORAL)
ax.set_title("FP rate vs items inserted\n(capacity=50K, target FP=1%)",
             fontweight="bold", color="#1F4E79")
ax.tick_params(axis="y", labelcolor=CORAL)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# plot 3: memory savings ratio
ax = axes[1, 0]
ratios = [s/b for s, b in zip(set_memory, bloom_memory)]
bars   = ax.bar(range(len(capacities)), ratios, color=TEAL,
                zorder=3, edgecolor="white")
ax.set_xticks(range(len(capacities)))
ax.set_xticklabels([f"{c//1000}K" for c in capacities])
ax.set_xlabel("Capacity (URLs)")
ax.set_ylabel("Memory savings ratio")
ax.set_title("How many times smaller is Bloom filter vs set",
             fontweight="bold", color="#1F4E79")
for bar, ratio in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{ratio:.0f}x", ha="center", fontsize=10,
            fontweight="bold", color="#085041")

# plot 4: summary stats
ax = axes[1, 1]
ax.axis("off")
summary = [
    ("Bloom filter parameters", ""),
    ("  Capacity",               "10,000 URLs"),
    ("  Target FP rate",         "1.00%"),
    ("  Empirical FP rate",      f"{empirical_fp:.3%}"),
    ("  Theoretical FP rate",    f"{bf.current_fp_rate():.3%}"),
    ("", ""),
    ("Performance", ""),
    ("  Insert speed",           f"{insert_time:.2f} µs/URL"),
    ("  Lookup speed",           f"{lookup_time:.2f} µs/URL"),
    ("", ""),
    ("Memory at 1M URLs", ""),
    ("  Bloom filter",           f"{bloom_memory[-1]:.2f} MB"),
    ("  Python set()",           f"{set_memory[-1]:.2f} MB"),
    ("  Savings",                f"{ratios[-1]:.0f}x smaller"),
    ("", ""),
    ("Hash functions (k)",       str(bf.k)),
    ("Bit array size (m)",       f"{bf.m:,} bits"),
]
y = 0.97
for label, val in summary:
    if val == "" and label:
        ax.text(0.02, y, label, transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="#1F4E79",
                verticalalignment="top")
    elif label:
        ax.text(0.02, y, label, transform=ax.transAxes,
                fontsize=9, color="#444", verticalalignment="top")
        ax.text(0.65, y, val, transform=ax.transAxes,
                fontsize=9, color="#1F4E79", fontweight="bold",
                verticalalignment="top")
    y -= 0.054
ax.set_title("Summary statistics", fontweight="bold", color="#1F4E79")

plt.tight_layout()
out = "experiments/graphs/bloom_filter_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nGraph saved: {out}")
plt.show()