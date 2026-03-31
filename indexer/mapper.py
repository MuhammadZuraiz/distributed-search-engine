"""
Phase 5 — Mapper
=================
Reads crawled JSON documents and emits (word, doc_id) pairs.
Each mapper process handles a partition of the documents.

Called by indexer/run_indexing.py — not run directly.
"""

import json
import os
import re


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "this", "that", "was",
    "are", "be", "as", "not", "have", "has", "he", "she", "they", "we",
    "you", "i", "do", "did", "will", "would", "could", "should", "may",
    "all", "no", "so", "if", "about", "up", "out", "into", "than", "then",
    "there", "their", "what", "which", "who", "more", "also", "been", "had"
}


def tokenise(text):
    """Lowercase, strip punctuation, remove stopwords and short tokens."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def map_document(filepath):
    """
    Read one crawled JSON file and return a list of (word, doc_id) pairs.
    This is the MAP phase output for one document.
    """
    with open(filepath, encoding="utf-8") as f:
        doc = json.load(f)

    doc_id = doc["doc_id"]
    words  = tokenise(doc.get("title", "") + " " + doc.get("text", ""))

    # emit (word, doc_id) for every word in this document
    pairs = [(word, doc_id) for word in words]
    return pairs


def map_partition(filepaths):
    """
    Run map phase over a list of files (one partition).
    Returns all (word, doc_id) pairs from that partition.
    """
    all_pairs = []
    for fp in filepaths:
        try:
            all_pairs.extend(map_document(fp))
        except Exception as e:
            print(f"[mapper] error on {fp}: {e}")
    return all_pairs\

# in indexer/mapper.py, add this alternative function:
def map_partition_from_db(doc_ids):
    """Map phase reading from SQLite instead of JSON files."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from db.database import get_document

    all_pairs = []
    for doc_id in doc_ids:
        try:
            doc   = get_document(doc_id)
            if not doc:
                continue
            words = tokenise(doc.get("title", "") + " " + doc.get("text", ""))
            all_pairs.extend((word, doc_id) for word in words)
        except Exception as e:
            print(f"[mapper] error on doc {doc_id}: {e}")
    return all_pairs