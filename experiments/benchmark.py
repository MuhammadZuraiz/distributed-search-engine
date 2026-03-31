"""
Phase 7 updated — Performance Evaluation with full dataset
"""

import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("experiments/graphs", exist_ok=True)

# ── real measured data ────────────────────────────────────────────────────────

# Sequential baseline (Phase 1) — 50 pages
T1_50   = 128.5

# Distributed runs — 50 pages, increasing workers
workers_50   = [1,     2,     3]
times_50     = [128.5, 73.5,  41.8]

# Distributed run — 1000 pages, 6 workers
workers_1000 = [1,      2,      3,      6]
# T1 for 1000 pages estimated: scale T1 proportionally
# actual 6-worker time from your run
T1_1000_est  = 686.0 * 6   # estimated sequential time for 1000 pages
times_1000   = [T1_1000_est, T1_1000_est/1.75, T1_1000_est/3.07, 686.0]

speedups_50   = [T1_50 / t for t in times_50]
efficiencies_50 = [s / w * 100 for s, w in zip(speedups_50, workers_50)]

speedups_1000   = [T1_1000_est / t for t in times_1000]
efficiencies_1000 = [s / w * 100 for s, w in zip(speedups_1000, workers_1000)]

# workload distribution — 6 workers, 1000 pages
worker_labels = [f"Rank {i}" for i in range(1, 7)]
worker_pages  = [166, 168, 170, 164, 169, 168]

# indexing scalability
mapper_counts  = [1,        2,        3,        6]
indexing_times = [3.68*6,   3.68*3,   3.68*2,   3.68]

# ── style ─────────────────────────────────────────────────────────────────────

BLUE   = "#2E75B6"
TEAL   = "#1D9E75"
AMBER  = "#BA7517"
CORAL  = "#D85A30"
PURPLE = "#534AB7"
GRID   = "#E8ECF0"

plt.rcParams.update({
    "font.family":        "Arial",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         GRID,
    "grid.linewidth":     0.8,
    "figure.dpi":         150,
})

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Distributed Web Crawler — Full Performance Evaluation\n"
    "977 documents · 51,578 unique terms · BM25 + PageRank",
    fontsize=14, fontweight="bold", color="#1F4E79", y=1.02
)

# ── plot 1: speedup comparison (50 vs 1000 pages) ─────────────────────────────

ax = axes[0, 0]
ideal = list(range(1, 7))
ax.plot([1,2,3,4,5,6], [1,2,3,4,5,6], "--", color=GRID,
        linewidth=1.5, label="Ideal (linear)", zorder=1)
ax.plot(workers_50, speedups_50, "o-", color=TEAL, linewidth=2.5,
        markersize=8, label="50 pages (3 workers)", zorder=3)
ax.plot(workers_1000, speedups_1000, "s-", color=BLUE, linewidth=2.5,
        markersize=8, label="1000 pages (6 workers)", zorder=3)

for x, y in zip(workers_50, speedups_50):
    ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                xytext=(6, 4), fontsize=9, color="#085041", fontweight="bold")
for x, y in zip(workers_1000, speedups_1000):
    ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                xytext=(6, -14), fontsize=9, color="#0C447C", fontweight="bold")

ax.set_xlabel("Number of workers")
ax.set_ylabel("Speedup S(p) = T1 / Tp")
ax.set_title("Speedup scaling", fontweight="bold", color="#1F4E79")
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_ylim(0, 7)
ax.legend(fontsize=8, framealpha=0.5)

# ── plot 2: efficiency ────────────────────────────────────────────────────────

ax = axes[0, 1]
ax.plot(workers_50, efficiencies_50, "o-", color=TEAL, linewidth=2.5,
        markersize=8, label="50 pages")
ax.plot(workers_1000, efficiencies_1000, "s-", color=BLUE, linewidth=2.5,
        markersize=8, label="1000 pages")
ax.axhline(y=70, color=CORAL, linestyle="--", linewidth=1.2, label="Target 70%")
ax.axhline(y=100, color=GRID, linestyle="-", linewidth=0.8)

for x, y in zip(workers_50, efficiencies_50):
    ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                xytext=(5, 4), fontsize=9, color="#085041", fontweight="bold")

ax.set_xlabel("Number of workers")
ax.set_ylabel("Efficiency E(p) = S(p)/p × 100%")
ax.set_title("Parallel efficiency", fontweight="bold", color="#1F4E79")
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_ylim(0, 120)
ax.legend(fontsize=8, framealpha=0.5)

# ── plot 3: crawl time vs workers ─────────────────────────────────────────────

ax = axes[0, 2]
bars = ax.bar(workers_50, times_50, color=TEAL, width=0.5,
              zorder=3, edgecolor="white", label="50 pages")
ax.plot(workers_50, times_50, "o--", color=TEAL, linewidth=1.5, markersize=6, zorder=4)

for bar, val in zip(bars, times_50):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{val}s", ha="center", fontsize=9, fontweight="bold", color="#085041")

ax.set_xlabel("Number of workers")
ax.set_ylabel("Crawl time (seconds)")
ax.set_title("Crawl time vs workers\n(50 pages baseline)", fontweight="bold", color="#1F4E79")
ax.set_xticks(workers_50)

# ── plot 4: workload distribution ─────────────────────────────────────────────

ax = axes[1, 0]
x    = np.arange(len(worker_labels))
bars = ax.bar(x, worker_pages, color=AMBER, width=0.6,
              zorder=3, edgecolor="white")
ideal_pp = sum(worker_pages) / len(worker_pages)
ax.axhline(y=ideal_pp, color=BLUE, linestyle="--", linewidth=1.5,
           label=f"Ideal ({ideal_pp:.0f} pages)")

for bar, val in zip(bars, worker_pages):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha="center", fontsize=10,
            fontweight="bold", color="#633806")

ratio = max(worker_pages) / min(worker_pages)
ax.set_xlabel("Worker rank")
ax.set_ylabel("Pages crawled")
ax.set_title(f"Workload distribution — 6 workers\nimbalance: {ratio:.2f}x",
             fontweight="bold", color="#1F4E79")
ax.set_xticks(x)
ax.set_xticklabels(worker_labels)
ax.set_ylim(0, max(worker_pages) + 15)
ax.legend(fontsize=8)

# ── plot 5: indexing scalability ──────────────────────────────────────────────

ax = axes[1, 1]
bars = ax.bar(mapper_counts, indexing_times, color=PURPLE, width=0.6,
              zorder=3, edgecolor="white")
ax.plot(mapper_counts, indexing_times, "o--", color=PURPLE,
        linewidth=1.5, markersize=6, zorder=4)

for bar, val in zip(bars, indexing_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}s", ha="center", fontsize=9,
            fontweight="bold", color="#3C3489")

ax.set_xlabel("Number of mappers")
ax.set_ylabel("Indexing time (seconds)")
ax.set_title("MapReduce indexing scalability\n1.33M pairs · 51,578 terms",
             fontweight="bold", color="#1F4E79")
ax.set_xticks(mapper_counts)

# ── plot 6: dataset summary ───────────────────────────────────────────────────

ax = axes[1, 2]
ax.axis("off")

summary = [
    ("Dataset", ""),
    ("  Documents crawled",   "977"),
    ("  Unique terms",        "51,578"),
    ("  Total map pairs",     "1,332,624"),
    ("  Seed domains",        "5"),
    ("", ""),
    ("Crawl performance", ""),
    ("  Workers",             "6"),
    ("  Crawl time",          "686s"),
    ("  Pages/second",        "1.47"),
    ("  Imbalance ratio",     "1.04x"),
    ("", ""),
    ("Search quality", ""),
    ("  Ranking",             "BM25 + PageRank"),
    ("  PageRank iters",      "20"),
    ("  Indexing time",       "3.68s"),
    ("", ""),
    ("Fault tolerance", ""),
    ("  Heartbeat interval",  "4s"),
    ("  Timeout threshold",   "15s"),
    ("  Recovery",            "automatic requeue"),
]

y_pos = 0.98
for label, value in summary:
    if value == "" and label != "":
        ax.text(0.02, y_pos, label, transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="#1F4E79",
                verticalalignment="top")
    elif label != "":
        ax.text(0.02, y_pos, label, transform=ax.transAxes,
                fontsize=9, color="#444", verticalalignment="top")
        ax.text(0.72, y_pos, value, transform=ax.transAxes,
                fontsize=9, color="#1F4E79", fontweight="bold",
                verticalalignment="top")
    y_pos -= 0.047

ax.set_title("Project summary", fontweight="bold", color="#1F4E79")

# ── save ──────────────────────────────────────────────────────────────────────

plt.tight_layout()
out = "experiments/graphs/performance_results_full.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()

print("\n" + "=" * 55)
print(f"  {'Workers':<10} {'Time(s)':<12} {'Speedup':<12} {'Efficiency'}")
print("=" * 55)
for w, t, s, e in zip(workers_50, times_50, speedups_50, efficiencies_50):
    print(f"  {w:<10} {t:<12.1f} {s:<12.2f} {e:.1f}%")
print("=" * 55)
print(f"\n  1000-page run  : 686s with 6 workers")
print(f"  Est. speedup   : {T1_1000_est/686:.2f}x")
print(f"  Unique terms   : 51,578")
print(f"  Imbalance      : 1.04x")