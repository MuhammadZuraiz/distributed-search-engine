"""
Semantic Search — TF-IDF vector cosine similarity
===================================================
Finds conceptually related documents even without
exact keyword matches. Blended with BM25 at query time.

Example: query "automobile speed" matches docs containing
"car velocity" because the vector spaces are similar.
"""

import json
import glob
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


VECTOR_CACHE = "data/index/tfidf_vectors.pkl"


def build_vectors(crawled_dir):
    """
    Build TF-IDF matrix from all crawled documents.
    Saves to disk so it only needs to be built once.
    Returns (vectorizer, matrix, doc_id_list)
    """
    print("[semantic] Building TF-IDF vectors...")
    files   = sorted(glob.glob(os.path.join(crawled_dir, "*.json")))
    corpus  = []
    doc_ids = []

    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
            text = doc.get("title", "") + " " + doc.get("text", "")
            corpus.append(text)
            doc_ids.append(doc["doc_id"])
        except Exception:
            continue

    vectorizer = TfidfVectorizer(
        max_features=30000,
        stop_words="english",
        ngram_range=(1, 2),      # unigrams + bigrams for better matching
        min_df=2,                # ignore terms appearing in only 1 doc
        sublinear_tf=True        # apply log normalization to TF
    )

    matrix = vectorizer.fit_transform(corpus)
    print(f"[semantic] Matrix shape: {matrix.shape} "
          f"({matrix.shape[0]} docs x {matrix.shape[1]} features)")

    # save to disk
    os.makedirs(os.path.dirname(VECTOR_CACHE), exist_ok=True)
    with open(VECTOR_CACHE, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "matrix":     matrix,
            "doc_ids":    doc_ids
        }, f)
    print(f"[semantic] Saved to {VECTOR_CACHE}")

    return vectorizer, matrix, doc_ids


def load_vectors():
    """Load cached TF-IDF vectors from disk."""
    if not os.path.exists(VECTOR_CACHE):
        return None, None, None
    with open(VECTOR_CACHE, "rb") as f:
        data = pickle.load(f)
    return data["vectorizer"], data["matrix"], data["doc_ids"]


class SemanticSearcher:
    def __init__(self, crawled_dir="data/crawled_distributed"):
        vec, mat, ids = load_vectors()
        if vec is None:
            vec, mat, ids = build_vectors(crawled_dir)

        self.vectorizer = vec
        self.matrix     = mat
        self.doc_ids    = ids
        print(f"[semantic] Ready — {len(ids)} document vectors loaded")

    def search(self, query, top_n=20):
        """
        Return list of (doc_id, cosine_score) sorted by similarity.
        """
        query_vec = self.vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, self.matrix).flatten()

        # get top_n indices
        top_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self.doc_ids[idx], score))

        return results