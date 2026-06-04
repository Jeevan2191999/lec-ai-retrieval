import json
import os
import random

def create_papers():
    random.seed(42)
    
    base_papers = [
        {"id": "001", "title": "Retrieval-Augmented Generation for Large Language Models", "text": "We survey retrieval-augmented generation (RAG) for large language models. RAG combines parametric knowledge with non-parametric retrieval to improve factual accuracy. We analyse three paradigms: naive RAG, advanced RAG, and modular RAG, covering retrieval, generation, and augmentation techniques."},
        {"id": "002", "title": "Dense Passage Retrieval for Open-Domain Question Answering", "text": "Open-domain question answering relies on efficient retrieval of relevant passages. We present DPR, using dense representations learned by dual-encoder BERT models. DPR outperforms BM25 by 9-19% in top-20 passage retrieval accuracy on multiple QA benchmarks."},
        {"id": "003", "title": "BM25 and Beyond: Lexical Retrieval in the Neural Era", "text": "BM25 remains a strong baseline for information retrieval despite neural advances. We analyse BM25 strengths in exact match scenarios and failure modes in semantic search. Hybrid approaches combining BM25 with dense retrievers consistently outperform either alone."},
        {"id": "004", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "text": "BERT obtains state-of-the-art results on eleven NLP tasks. Pre-training uses masked language modelling and next sentence prediction. Fine-tuning requires only one additional output layer, making it applicable across question answering and inference."},
        {"id": "005", "title": "Attention Is All You Need", "text": "The Transformer architecture relies entirely on attention mechanisms, dispensing with recurrence and convolutions. Multi-head attention allows the model to attend to information from different representation subspaces. The model achieves state-of-the-art on machine translation with significantly less training time."},
        {"id": "006", "title": "Hybrid Retrieval for Question Answering", "text": "Combining sparse BM25 and dense retrieval improves recall across diverse query types. We study linear interpolation and learned fusion strategies. Hybrid retrieval reduces failure cases where dense models struggle with rare terminology."},
        {"id": "007", "title": "ColBERT: Efficient and Effective Passage Search", "text": "ColBERT introduces late interaction for scalable semantic retrieval. Token-level embeddings are compressed and indexed for fast MaxSim scoring. ColBERT achieves high MRR while maintaining sub-millisecond query latency through approximate nearest neighbour search."},
        {"id": "008", "title": "FAISS: A Library for Efficient Similarity Search", "text": "FAISS provides efficient similarity search for dense vectors at billion scale. It supports exact and approximate nearest neighbour search with GPU acceleration. Product quantisation reduces memory footprint while maintaining retrieval quality above 95% recall."},
        {"id": "009", "title": "Sentence-BERT: Sentence Embeddings using Siamese Networks", "text": "SBERT modifies BERT using siamese and triplet networks to produce semantically meaningful sentence embeddings. Cosine similarity between embeddings correlates with semantic similarity. SBERT reduces inference time from 65 hours to 5 seconds for 10,000 sentence pairs."},
        {"id": "010", "title": "Multi-Hop Question Answering over Knowledge Graphs", "text": "Multi-hop reasoning requires combining evidence across multiple documents. We propose a graph-based retrieval approach that identifies reasoning paths. Our method outperforms single-hop retrievers on HotpotQA and 2WikiMultiHop by 12% and 15% respectively."},
        {"id": "011", "title": "LLaMA: Open and Efficient Foundation Language Models", "text": "LLaMA provides open foundation models from 7B to 65B parameters trained on public data. Smaller models trained longer outperform larger models trained less. LLaMA-13B outperforms GPT-3 on most benchmarks despite being 10x smaller."},
        {"id": "012", "title": "Chain-of-Thought Prompting Elicits Reasoning", "text": "Chain-of-thought prompting improves complex reasoning in large language models. Providing exemplars with reasoning steps enables models to decompose problems. Performance on arithmetic, commonsense, and symbolic reasoning improves significantly."},
        {"id": "013", "title": "In-Context Learning with Retrieval Augmentation", "text": "Combining in-context learning with retrieved examples improves few-shot performance. Retrieved demonstrations are more relevant than randomly selected ones. We study BM25, dense, and diversified retrieval strategies for in-context learning."},
        {"id": "014", "title": "Zero-Shot Question Generation for Retrieval", "text": "Generating pseudo-questions for passages improves dense retrieval training. Zero-shot question generation with large language models creates diverse training signal. Our approach improves recall@10 by 8% on BEIR benchmarks without requiring labelled data."},
        {"id": "015", "title": "Efficient Transformers: A Survey", "text": "Transformer attention scales quadratically with sequence length. We survey efficient attention mechanisms including sparse attention, linear attention, and memory-augmented approaches. Efficient transformers enable processing of longer documents critical for retrieval."},
        {"id": "016", "title": "REALM: Retrieval-Augmented Language Model Pre-Training", "text": "REALM integrates retrieval into language model pre-training. The retriever and language model are jointly trained end-to-end. REALM improves open-domain QA accuracy by learning to retrieve useful background knowledge during pre-training."},
        {"id": "017", "title": "Contrastive Learning for Dense Retrieval", "text": "Contrastive learning with hard negatives improves dense retrieval quality. In-batch negatives and mined hard negatives provide complementary training signal. Our approach improves MRR@10 by 4 points on MS MARCO compared to standard fine-tuning."},
        {"id": "018", "title": "Neural Reranking for Information Retrieval", "text": "Cross-encoder rerankers improve precision over first-stage retrievers. BERT-based rerankers score query-passage pairs jointly. Two-stage retrieval with reranking achieves higher MRR than single-stage retrieval while maintaining practical latency."},
        {"id": "019", "title": "Knowledge-Intensive Language Tasks Benchmark", "text": "KILT provides a unified benchmark for knowledge-intensive NLP tasks including fact verification, entity linking, slot filling, open-domain QA, and dialogue. Retrieval quality strongly correlates with downstream task performance across all categories."},
        {"id": "020", "title": "Faithfulness in Retrieval Augmented Generation", "text": "RAG systems may generate answers not supported by retrieved passages. Faithfulness evaluation measures overlap between generated answers and source passages. Post-hoc verification using natural language inference improves factual consistency significantly."},
        {"id": "021", "title": "Vector Databases for Semantic Search at Scale", "text": "Vector databases store dense embeddings for efficient similarity search. Systems like Pinecone, Weaviate, and Chroma support approximate nearest neighbour queries. Hybrid filtering combines metadata filters with vector search for production retrieval systems."},
        {"id": "022", "title": "Sparse vs Dense Retrieval Trade-offs", "text": "Sparse retrievers excel at exact keyword matching while dense retrievers capture semantics. Performance varies significantly across domains and query types. Ensemble methods combining both consistently outperform either alone across diverse benchmarks."},
        {"id": "023", "title": "Long Document Question Answering", "text": "Long document QA requires hierarchical or sliding window approaches. Dense retrieval struggles with very long documents due to truncation. Recursive summarisation and chunking strategies improve recall on book and report QA benchmarks."},
        {"id": "024", "title": "Evaluation Metrics for Information Retrieval Systems", "text": "Standard IR metrics include MRR, NDCG, MAP, and Recall@K. Mean Reciprocal Rank measures the average rank of the first relevant result. Normalised Discounted Cumulative Gain accounts for position and graded relevance in evaluation."},
        {"id": "025", "title": "Domain Adaptation for Neural Retrieval Models", "text": "General-purpose retrievers underperform on specialised domains. Fine-tuning on domain-specific data improves retrieval significantly. Unsupervised domain adaptation using contrastive learning reduces labelling requirements for new domains."},
        {"id": "026", "title": "Conversational Search and Query Rewriting", "text": "Conversational queries contain coreference and ellipsis requiring resolution. Query rewriting using dialogue history improves retrieval recall. We study automatic rewriting with T5 and GPT-2 for conversational passage retrieval benchmarks."},
        {"id": "027", "title": "Cross-Lingual Information Retrieval", "text": "Cross-lingual retrieval matches queries and documents in different languages. Multilingual encoders like mBERT and XLM-R enable zero-shot cross-lingual transfer. We evaluate on CLEF and MLIA benchmarks across 15 language pairs."},
        {"id": "028", "title": "Passage Segmentation Strategies for Retrieval", "text": "Document chunking strategy significantly impacts retrieval quality. Fixed-size, sentence-based, and semantic chunking produce different precision-recall trade-offs. Overlapping chunks with 20% overlap improve recall for queries spanning chunk boundaries."},
        {"id": "029", "title": "Temporal Reasoning in Open-Domain Question Answering", "text": "Knowledge cutoffs cause retrievers to return outdated information. Time-aware retrieval weights recent documents higher for temporal queries. We study decay functions and temporal filtering strategies for news and encyclopaedia corpora."},
        {"id": "030", "title": "Prompt Engineering for Retrieval Augmented Generation", "text": "Prompt design significantly affects RAG output quality. We study instruction formats, context placement, and citation requirements. Structured prompts with explicit grounding instructions improve faithfulness by 15% on TriviaQA and Natural Questions."},
    ]

    papers = list(base_papers)
    
    # Expand to 300 papers
    while len(papers) < 300:
        template = random.choice(base_papers)
        idx = len(papers) + 1
        papers.append({
            "id": f"{idx:03d}",
            "title": f"{template['title']} - Extended Study {idx}",
            "text": template["text"],
            "abstract": template["text"],
            "categories": ["cs.IR"],
            "authors": ["Anonymous"]
        })
    
    os.makedirs("data", exist_ok=True)
    with open("data/papers.json", "w") as f:
        json.dump(papers, f, indent=2)
    
    print(f"Created {len(papers)} papers successfully!")
    return papers

if __name__ == "__main__":
    create_papers()