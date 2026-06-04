import json
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def recall_at_k(results, relevant_ids, k=5):
    retrieved_ids = [r["paper"]["id"] for r in results[:k]]
    hits = sum(1 for rid in relevant_ids if rid in retrieved_ids)
    return hits / len(relevant_ids)

def reciprocal_rank(results, relevant_ids):
    for rank, r in enumerate(results, start=1):
        if r["paper"]["id"] in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retriever(retriever, queries, k=5):
    recall_scores = []
    rr_scores = []
    latencies = []

    for q in queries:
        results, latency = retriever.search(q["query"], top_k=k)
        recall_scores.append(recall_at_k(results, q["relevant_ids"], k))
        rr_scores.append(reciprocal_rank(results, q["relevant_ids"]))
        latencies.append(latency)

    return {
        "recall_at_5": round(np.mean(recall_scores), 3),
        "mrr": round(np.mean(rr_scores), 3),
        "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        "mean_latency_ms": round(np.mean(latencies), 2)
    }

if __name__ == "__main__":
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.hybrid_retriever import HybridRetriever

    with open("data/papers.json") as f:
        papers = json.load(f)

    with open("queries/queries.json") as f:
        queries = json.load(f)

    print("Evaluating BM25...")
    bm25 = BM25Retriever(papers)
    bm25_results = evaluate_retriever(bm25, queries)
    print(f"BM25:   Recall@5={bm25_results['recall_at_5']} | MRR={bm25_results['mrr']} | p95={bm25_results['p95_latency_ms']}ms")

    print("\nEvaluating Dense...")
    dense = DenseRetriever(papers)
    dense_results = evaluate_retriever(dense, queries)
    print(f"Dense:  Recall@5={dense_results['recall_at_5']} | MRR={dense_results['mrr']} | p95={dense_results['p95_latency_ms']}ms")

    print("\nEvaluating Hybrid...")
    hybrid = HybridRetriever(papers)
    hybrid_results = evaluate_retriever(hybrid, queries)
    print(f"Hybrid: Recall@5={hybrid_results['recall_at_5']} | MRR={hybrid_results['mrr']} | p95={hybrid_results['p95_latency_ms']}ms")

    print("\n=== FINAL RESULTS ===")
    print(f"{'Config':<10} {'Recall@5':<12} {'MRR':<10} {'p95 Latency'}")
    print(f"{'BM25':<10} {bm25_results['recall_at_5']:<12} {bm25_results['mrr']:<10} {bm25_results['p95_latency_ms']}ms")
    print(f"{'Dense':<10} {dense_results['recall_at_5']:<12} {dense_results['mrr']:<10} {dense_results['p95_latency_ms']}ms")
    print(f"{'Hybrid':<10} {hybrid_results['recall_at_5']:<12} {hybrid_results['mrr']:<10} {hybrid_results['p95_latency_ms']}ms")