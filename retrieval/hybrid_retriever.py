import json
import time
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever

class HybridRetriever:
    def __init__(self, papers, alpha=0.5):
        """
        alpha: weight for dense scores (0=pure BM25, 1=pure dense)
        """
        self.papers = papers
        self.alpha = alpha
        print("Building hybrid retriever...")
        self.bm25 = BM25Retriever(papers)
        self.dense = DenseRetriever(papers)
        print("Hybrid retriever ready!")

    def search(self, query, top_k=5):
        start = time.time()

        bm25_results, _ = self.bm25.search(query, top_k=len(self.papers))
        dense_results, _ = self.dense.search(query, top_k=len(self.papers))

        # Normalise BM25 scores 0-1
        bm25_scores = {}
        bm25_raw = [r["score"] for r in bm25_results]
        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1
        for r in bm25_results:
            bm25_scores[r["paper"]["id"]] = r["score"] / max_bm25

        # Normalise dense scores 0-1
        dense_scores = {}
        dense_raw = [r["score"] for r in dense_results]
        min_d, max_d = min(dense_raw), max(dense_raw)
        for r in dense_results:
            if max_d - min_d > 0:
                dense_scores[r["paper"]["id"]] = (r["score"] - min_d) / (max_d - min_d)
            else:
                dense_scores[r["paper"]["id"]] = 0.5

        # Combine scores
        all_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        combined = {}
        for pid in all_ids:
            b = bm25_scores.get(pid, 0)
            d = dense_scores.get(pid, 0)
            combined[pid] = (1 - self.alpha) * b + self.alpha * d

        # Sort by combined score
        top_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_k]

        latency = (time.time() - start) * 1000

        results = []
        paper_map = {p["id"]: p for p in self.papers}
        for rank, pid in enumerate(top_ids):
            results.append({
                "paper": paper_map[pid],
                "score": combined[pid],
                "rank": rank + 1
            })

        return results, latency


if __name__ == "__main__":
    with open("data/papers.json") as f:
        papers = json.load(f)

    retriever = HybridRetriever(papers, alpha=0.5)
    results, latency = retriever.search("transformer attention mechanism")

    print(f"\nLatency: {latency:.2f}ms")
    print("\nTop 5 results:")
    for r in results:
        print(f"Rank {r['rank']}: {r['paper']['title'][:70]}")
