import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class DenseRetriever:
    def __init__(self, papers):
        self.papers = papers
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Building dense index...")
        texts = [doc["text"] for doc in papers]
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalised = self.embeddings / norms
        self.index.add(normalised.astype(np.float32))
        
        print(f"Dense index built with {len(papers)} documents")

    def search(self, query, top_k=5):
        start = time.time()
        
        query_embedding = self.model.encode([query])
        query_norm = np.linalg.norm(query_embedding)
        query_normalised = (query_embedding / query_norm).astype(np.float32)
        
        scores, indices = self.index.search(query_normalised, top_k)
        latency = (time.time() - start) * 1000
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            results.append({
                "paper": self.papers[idx],
                "score": float(score),
                "rank": rank + 1
            })
        
        return results, latency

if __name__ == "__main__":
    with open("data/papers.json") as f:
        papers = json.load(f)
    
    retriever = DenseRetriever(papers)
    results, latency = retriever.search("transformer attention mechanism")
    
    print(f"\nLatency: {latency:.2f}ms")
    print("\nTop 5 results:")
    for r in results:
        print(f"Rank {r['rank']}: {r['paper']['title'][:70]}")
        