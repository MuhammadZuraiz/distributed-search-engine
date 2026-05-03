"""
Phase 5 — Reducer
==================
Takes (word, doc_id) pairs and builds the inverted index.
Also stores term frequency (TF) per document for basic ranking.

Called by indexer/run_indexing.py — not run directly.
"""

import json
import os
from collections import defaultdict


def reduce_pairs(pairs):
    """
    Reduce phase: group (word, doc_id) pairs into an inverted index.

    Output structure:
        {
          "word": {
              "doc_ids": [0, 3, 7],
              "tf": {"0": 4, "3": 1, "7": 2}   <- term frequency per doc
          },
          ...
        }
    """
    # count term frequency per (word, doc_id)
    tf_counts = defaultdict(lambda: defaultdict(int))
    for word, doc_id in pairs:
        tf_counts[word][doc_id] += 1

    # build final index
    index = {}
    for word, doc_freqs in tf_counts.items():
        index[word] = {
            "doc_ids": sorted(doc_freqs.keys()),
            "tf":      {str(d): c for d, c in doc_freqs.items()}
        }

    return index


def save_index(index, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"[reducer] index saved: {len(index)} unique terms -> {output_path}")


def merge_indexes(index_a, index_b):
    """Merge two partial indexes (used when running multiple reducers)."""
    merged = dict(index_a)
    for word, data in index_b.items():
        if word in merged:
            existing_ids = set(merged[word]["doc_ids"])
            new_ids      = set(data["doc_ids"])
            merged_ids   = sorted(existing_ids | new_ids)
            merged_tf    = dict(merged[word]["tf"])
            for doc_str, count in data["tf"].items():
                merged_tf[doc_str] = merged_tf.get(doc_str, 0) + count
            merged[word] = {"doc_ids": merged_ids, "tf": merged_tf}
        else:
            merged[word] = data
    return merged