"""
Incremental indexing — index only new documents
=================================================
Tracks which doc_ids are already indexed in SQLite.
Only processes the delta (new docs since last index run).
Merges delta index into main index without full re-index.

This means adding 50 new pages costs O(50) not O(1000).
"""

import json
import os
from collections import defaultdict

INDEX_PATH = "data/index/inverted_index.json"


def load_main_index():
    """Load existing inverted index from disk."""
    if not os.path.exists(INDEX_PATH):
        return {}
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


def merge_into_main(main_index, delta_index):
    """
    Merge a delta index into the main index in-place.
    For each term in delta:
      - if term exists in main: merge doc_ids and tf counts
      - if term is new: add it directly
    Returns updated main index.
    """
    for term, delta_entry in delta_index.items():
        if term not in main_index:
            main_index[term] = delta_entry
        else:
            existing     = main_index[term]
            existing_ids = set(existing["doc_ids"])
            new_ids      = set(delta_entry["doc_ids"])
            merged_ids   = sorted(existing_ids | new_ids)

            merged_tf    = dict(existing["tf"])
            for doc_str, count in delta_entry["tf"].items():
                merged_tf[doc_str] = merged_tf.get(doc_str, 0) + count

            main_index[term] = {
                "doc_ids": merged_ids,
                "tf":      merged_tf
            }

    return main_index


def remove_from_index(main_index, doc_ids_to_remove):
    """
    Remove a set of doc_ids from the index.
    Used when documents are re-crawled and need re-indexing.
    """
    doc_ids_set  = {str(d) for d in doc_ids_to_remove}
    int_ids_set  = set(doc_ids_to_remove)
    terms_to_del = []

    for term, entry in main_index.items():
        new_ids = [d for d in entry["doc_ids"] if d not in int_ids_set]
        new_tf  = {k: v for k, v in entry["tf"].items()
                   if k not in doc_ids_set}
        if not new_ids:
            terms_to_del.append(term)
        else:
            main_index[term] = {"doc_ids": new_ids, "tf": new_tf}

    for term in terms_to_del:
        del main_index[term]

    return main_index


def run_incremental_index():
    """
    Main entry point for incremental indexing.
    1. Find unindexed doc_ids from SQLite
    2. Map only those documents
    3. Merge delta into main index
    4. Mark them as indexed
    5. Update PageRank only if link graph changed significantly
    """
    from db.database import (get_unindexed_doc_ids, get_document,
                             mark_indexed, update_pagerank)
    from indexer.mapper import tokenise
    from indexer.reducer import reduce_pairs, save_index

    unindexed = get_unindexed_doc_ids()

    if not unindexed:
        print("[incremental] No new documents to index.")
        return 0

    print(f"[incremental] Indexing {len(unindexed)} new documents...")

    # ── map phase (sequential for small deltas) ───────────────────────────
    all_pairs = []
    for doc_id in unindexed:
        doc = get_document(doc_id)
        if not doc:
            continue
        words = tokenise(doc.get("title", "") + " " + doc.get("text", ""))
        pairs = [(word, doc_id) for word in words]
        all_pairs.extend(pairs)

    print(f"[incremental] Delta map pairs: {len(all_pairs):,}")

    # ── reduce phase ──────────────────────────────────────────────────────
    from indexer.reducer import reduce_pairs
    delta_index = reduce_pairs(all_pairs)
    print(f"[incremental] Delta unique terms: {len(delta_index):,}")

    # ── merge into main index ─────────────────────────────────────────────
    main_index  = load_main_index()
    before_size = len(main_index)
    main_index  = merge_into_main(main_index, delta_index)
    after_size  = len(main_index)

    save_index(main_index, INDEX_PATH)
    print(f"[incremental] Index: {before_size} -> {after_size} terms "
          f"(+{after_size - before_size} new)")

    # ── mark as indexed ───────────────────────────────────────────────────
    for doc_id in unindexed:
        word_count = sum(1 for _, d in all_pairs if d == doc_id)
        mark_indexed(doc_id, word_count)

    # ── update PageRank if many new docs ──────────────────────────────────
    threshold = 50
    if len(unindexed) >= threshold:
        print(f"[incremental] {len(unindexed)} new docs — updating PageRank...")
        from indexer.pagerank import compute_pagerank
        pr_scores = compute_pagerank("data/crawled_distributed")
        update_pagerank(pr_scores)
        pr_path = "data/index/pagerank.json"
        with open(pr_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in pr_scores.items()},
                      f, ensure_ascii=False)
        print(f"[incremental] PageRank updated: {len(pr_scores)} scores")
    else:
        print(f"[incremental] Skipping PageRank update "
              f"({len(unindexed)} < {threshold} threshold)")

    print(f"[incremental] Done. {len(unindexed)} documents indexed.")
    return len(unindexed)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    count = run_incremental_index()
    print(f"Indexed {count} new documents")