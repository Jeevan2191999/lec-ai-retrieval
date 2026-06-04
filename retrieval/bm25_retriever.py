import json
import time
import numpy as np
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, papers):
        self.papers = papers
        self.tokenised = [
            doc["text"].lower().split()
            for doc in papers
        ]
        self.bm25 = BM25Okapi(self.tokenised)
        print(f"BM25 index built with {len(papers)} documents")

    def search(self, query, top_k=5):
        start = time.time()
        tokenised_query = query.lower().split()
        scores = self.bm25.get_scores(tokenised_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        latency = (time.time() - start) * 1000

        results = []
        for idx in top_indices:
            results.append({
                "paper": self.papers[idx],
                "score": float(scores[idx]),
                "rank": len(results) + 1
            })

        return results, latency

if __name__ == "__main__":
    with open("data/papers.json") as f:
        papers = json.load(f)

    retriever = BM25Retriever(papers)
    results, latency = retriever.search("transformer attention mechanism")

    print(f"\nLatency: {latency:.2f}ms")
    print("\nTop 5 results:")
    for r in results:
        print(f"Rank {r['rank']}: {r['paper']['title'][:70]}")
        