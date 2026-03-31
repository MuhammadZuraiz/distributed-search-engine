"""
BERT embeddings + FAISS approximate nearest neighbour search
=============================================================
Replaces TF-IDF cosine similarity with sentence-transformer
embeddings (all-MiniLM-L6-v2, 384 dimensions).

This gives true semantic understanding:
  - "king - man + woman" ≈ "queen"
  - "fast car" finds docs about "rapid automobile"
  - "distributed fault" finds docs about "node failure recovery"

First run downloads ~80MB model. Subsequent runs load from cache.
"""

import os
import json
import pickle
import numpy as np

EMBEDDINGS_CACHE = "data/index/bert_embeddings.pkl"
FAISS_INDEX_PATH = "data/index/faiss.index"
DOCIDS_PATH      = "data/index/faiss_doc_ids.json"
MODEL_NAME       = "all-MiniLM-L6-v2"


def build_embeddings(crawled_dir=None):
    """
    Build BERT embeddings for all documents in SQLite.
    Saves FAISS index and doc_id mapping to disk.
    """
    import faiss
    from sentence_transformers import SentenceTransformer
    from db.database import get_all_documents

    print(f"[bert] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("[bert] Loading documents from DB...")
    docs    = get_all_documents()
    texts   = []
    doc_ids = []

    for doc in docs:
        title   = doc.get("title", "") or ""
        snippet = doc.get("snippet", "") or ""
        # use title + snippet for embedding (faster than full text)
        # title gets 3x weight by repetition
        text = f"{title} {title} {title} {snippet}"
        texts.append(text[:512])    # cap at 512 chars
        doc_ids.append(doc["doc_id"])

    print(f"[bert] Encoding {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True   # L2 norm for cosine similarity via dot product
    )

    print(f"[bert] Building FAISS index ({embeddings.shape})...")
    dim   = embeddings.shape[1]    # 384 for MiniLM
    index = faiss.IndexFlatIP(dim) # Inner product = cosine sim (after L2 norm)
    index.add(embeddings.astype(np.float32))

    # save everything
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(DOCIDS_PATH, "w") as f:
        json.dump(doc_ids, f)

    # cache the model separately for reuse
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump({"model_name": MODEL_NAME, "dim": dim,
                     "doc_count": len(doc_ids)}, f)

    print(f"[bert] Done — {len(doc_ids)} docs, {dim}d embeddings")
    print(f"[bert] FAISS index saved: {FAISS_INDEX_PATH}")
    return index, doc_ids, model


def load_embeddings():
    """Load FAISS index and doc_ids from disk. Returns (index, doc_ids, model) or None."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCIDS_PATH):
        return None, None, None
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        index   = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCIDS_PATH) as f:
            doc_ids = json.load(f)
        model = SentenceTransformer(MODEL_NAME)
        print(f"[bert] Loaded FAISS index: {index.ntotal} vectors, "
              f"{len(doc_ids)} doc_ids")
        return index, doc_ids, model
    except Exception as e:
        print(f"[bert] Load failed: {e}")
        return None, None, None


class BERTSearcher:
    def __init__(self, crawled_dir="data/crawled_distributed"):
        index, doc_ids, model = load_embeddings()
        if index is None:
            print("[bert] No cached index found — building now...")
            index, doc_ids, model = build_embeddings(crawled_dir)

        self.index   = index
        self.doc_ids = doc_ids
        self.model   = model
        print(f"[bert] Ready — {len(doc_ids)} document embeddings")

    def search(self, query, top_n=20):
        """
        Encode query and find top_n nearest documents.
        Returns list of (doc_id, score) sorted by similarity.
        """
        import numpy as np
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_n)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc_id = self.doc_ids[idx]
            results.append((doc_id, float(score)))

        return results

    def rebuild(self):
        """Rebuild index from current DB contents."""
        self.index, self.doc_ids, self.model = build_embeddings()