"""
Phase 5 — Parallel MapReduce Indexing Orchestrator (fixed)
"""

import glob
import json
import os
import sys
import time

from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from indexer.mapper import map_partition_from_db
from indexer.reducer import reduce_pairs, save_index
from utils.logger    import get_logger

CRAWLED_DIR  = "data/crawled_distributed"
INDEX_PATH   = "data/index/inverted_index.json"
URL_MAP_PATH = "data/index/url_map.json"

log = get_logger("indexer")

# replace build_url_map function with:
def build_url_map_from_db():
    from db.database import get_url_map
    return get_url_map()

def run():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ── every rank computes its partition ─────────────────────────────────────
    if rank == 0:
        my_pairs = []     # rank 0 is reducer only, no mapping
        start    = time.time()
        log.info(f"MapReduce indexing started — {size-1} mapper(s)")
    else:
        from db.database import get_document_count, get_conn
        conn     = get_conn()
        all_ids  = [r[0] for r in conn.execute(
                    "SELECT doc_id FROM documents ORDER BY doc_id").fetchall()]
        num_mappers = size - 1
        my_ids      = all_ids[rank - 1 :: num_mappers]

        log.info(f"Mapper {rank}: processing {len(my_ids)} documents from DB")
        my_pairs = map_partition_from_db(my_ids)
        log.info(f"Mapper {rank}: emitting {len(my_pairs):,} pairs")

    # ── gather all pairs to rank 0 in one collective call ─────────────────────
    # gather() blocks until ALL ranks call it — no deadlock possible
    all_gathered = comm.gather(my_pairs, root=0)

    # ── rank 0: reduce + save ─────────────────────────────────────────────────
    if rank == 0:
        all_pairs = []
        for i, pairs in enumerate(all_gathered):
            if pairs:
                log.info(f"  received {len(pairs):,} pairs from mapper {i}")
                all_pairs.extend(pairs)

        log.info(f"Total pairs: {len(all_pairs):,} — running reduce...")
        index = reduce_pairs(all_pairs)
        save_index(index, INDEX_PATH)

        # ── PageRank from DB ──────────────────────────────────────────────
        log.info("Computing PageRank...")
        from indexer.pagerank import build_link_graph, compute_pagerank
        from db.database import get_all_documents, update_pagerank

        # build link graph from DB instead of JSON files
        docs     = {d["doc_id"]: d for d in get_all_documents()}
        url_to_id = {d["url"]: d["doc_id"] for d in docs.values()}

        # run PageRank directly on DB data
        from collections import defaultdict
        DAMPING    = 0.85
        ITERATIONS = 20
        n          = len(docs)
        all_ids    = list(docs.keys())
        links_out  = defaultdict(list)
        links_in   = defaultdict(list)

        for src_id, doc in docs.items():
            seen = set()
            for target_url in doc.get("links", []):
                tgt_id = url_to_id.get(target_url)
                if tgt_id and tgt_id != src_id and tgt_id not in seen:
                    links_out[src_id].append(tgt_id)
                    links_in[tgt_id].append(src_id)
                    seen.add(tgt_id)

        scores = {doc_id: 1.0 / n for doc_id in all_ids}
        for iteration in range(ITERATIONS):
            new_scores = {}
            for doc_id in all_ids:
                incoming = sum(
                    scores[src] / len(links_out[src])
                    for src in links_in.get(doc_id, [])
                    if links_out.get(src)
                )
                new_scores[doc_id] = (1 - DAMPING) / n + DAMPING * incoming
            delta  = sum(abs(new_scores[d] - scores[d]) for d in all_ids)
            scores = new_scores
            if delta < 1e-6:
                log.info(f"PageRank converged at iteration {iteration + 1}")
                break

        max_score = max(scores.values()) or 1.0
        pr_scores = {doc_id: round(s / max_score, 6)
                     for doc_id, s in scores.items()}

        # save to DB and JSON
        update_pagerank(pr_scores)
        pr_path = "data/index/pagerank.json"
        os.makedirs(os.path.dirname(pr_path), exist_ok=True)
        with open(pr_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in pr_scores.items()},
                      f, ensure_ascii=False)
        log.info(f"PageRank saved: {len(pr_scores)} scores")

        # ── BERT embeddings + FAISS ───────────────────────────────────────
        log.info("Building BERT embeddings + FAISS index...")
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from search.embeddings import build_embeddings
        build_embeddings()
        log.info("BERT embeddings built and cached")

        # ── TF-IDF semantic vectors (kept as fallback) ────────────────────
        log.info("Building TF-IDF fallback vectors...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        import pickle as pkl

        corpus  = []
        doc_ids_list = []
        for doc in get_all_documents():
            corpus.append(doc.get("title","") + " " + doc.get("text",""))
            doc_ids_list.append(doc["doc_id"])

        vectorizer = TfidfVectorizer(
            max_features=30000, stop_words="english",
            ngram_range=(1,2), min_df=2, sublinear_tf=True
        )
        matrix = vectorizer.fit_transform(corpus)
        VECTOR_CACHE = "data/index/tfidf_vectors.pkl"
        with open(VECTOR_CACHE, "wb") as f:
            pkl.dump({"vectorizer": vectorizer,
                      "matrix": matrix, "doc_ids": doc_ids_list}, f)
        log.info(f"TF-IDF matrix: {matrix.shape}")

        # ── URL map from DB ───────────────────────────────────────────────
        url_map = build_url_map_from_db()
        os.makedirs(os.path.dirname(URL_MAP_PATH), exist_ok=True)
        with open(URL_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(url_map, f, ensure_ascii=False, indent=2)

        # ── mark all docs as indexed ──────────────────────────────────────
        from db.database import mark_indexed
        for doc_id in doc_ids_list:
            mark_indexed(doc_id, 0)

        elapsed = time.time() - start
        log.info("-" * 60)
        log.info(f"Indexing complete in {elapsed:.2f}s")
        log.info(f"Unique terms : {len(index):,}")
        log.info(f"Documents    : {len(url_map)}")
        top5 = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        log.info(f"Top PageRank : {top5}")
        log.info("-" * 60)


if __name__ == "__main__":
    run()