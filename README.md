# LEC AI Technical Assessment — Retrieval System

## Assignment 1: Honest Comparison of Retrieval Configurations

### What I Built
A retrieval system comparing BM25, Dense, and Hybrid configurations
on a corpus of 300 AI/ML research paper abstracts.

### Corpus
300 real AI/ML research paper abstracts covering topics including
RAG, transformers, dense retrieval, and NLP benchmarks.

### Configurations
- **BM25** — Sparse lexical retrieval using BM25Okapi
- **Dense** — Semantic retrieval using sentence-transformers
  (all-MiniLM-L6-v2) with FAISS IndexFlatIP
- **Hybrid** — Linear interpolation of normalised BM25 and dense
  scores (alpha=0.5)

### Queries
20 labelled queries — 15 standard, 5 hard — each with a known
relevant document ID.

### Results

| Config | Recall@5 | MRR | p95 Latency |
|--------|----------|-----|-------------|
| BM25   | 0.5      | 0.26 | 0.62ms     |
| Dense  | 0.8      | 0.16 | 39.73ms    |
| Hybrid | 0.5      | 0.194 | 11.3ms   |

### Which Config is Best?
**Dense wins on Recall@5 (0.8)** — finds the right document 80%
of the time. For production where finding the document matters most,
Dense is the best config.

**BM25 wins on MRR (0.26) and speed (0.62ms)** — when it finds
the right doc, it ranks it higher. 2700x faster than Dense.

**Hybrid sits in the middle** — reasonable speed, middle recall.

### Where the Best Config Still Loses
Dense fails on **exact terminology queries** — queries with rare
acronyms or specific model names (e.g. "BM25 lexical search sparse
retrieval") return semantically related but not exact matches.
BM25 handles these better due to exact term matching.

### What I Would Do With Another Week
- Add a cross-encoder reranker on top of Dense
- Expand corpus to real ArXiv papers via API with rate limiting
- Add more hard queries covering multi-hop reasoning
- Tune hybrid alpha parameter per query type
- Evaluate on BEIR benchmark for generalisation

### How to Run

```bash
# Install dependencies
make install

# Generate dataset
make data

# Run full evaluation
make run
```