run:
	python3 main.py

data:
	python3 data/fetch_arxiv.py

eval:
	python3 evaluation/evaluator.py

install:
	pip install arxiv rank_bm25 sentence-transformers faiss-cpu numpy pandas tqdm