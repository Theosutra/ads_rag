# Tableau des hyperparamètres — Mémoire RAG ADS CESI

| Composant | Paramètre | Valeurs testées | Valeur retenue |
|-----------|-----------|-----------------|----------------|
| BM25 | k1 | 0.9, 1.0, 1.2 | **1.0** |
| BM25 | b | 0.4, 0.6, 0.75 | **0.6** |
| Chunking | taille (tokens) | 128, 200, 256, 300, 400 | **256** |
| Chunking | overlap (tokens) | 0, 32, 50, 64 | **50** |
| Retrieval | k (top-k) | 5, 10, 20, 50 | **5, 10, 20** (comparés) |
| Embeddings FR | modèle | multilingual-e5-large, LaBSE | **intfloat/multilingual-e5-large** |
| Embeddings EN | modèle | DPR, SBERT | **facebook/dpr-ctx_encoder-single-nq-base** |
| Hybride | α (fusion) | 0.3, 0.5, 0.7 | **0.5** |
| Générateur | modèle | google/mt5-large | **google/mt5-large** |
| Générateur | max_new_tokens | 64, 128, 256 | **128** |
| Générateur | num_beams | 1, 4 | **4** |
| RAGAS | LLM juge | gpt-3.5-turbo, llama-3-8b | **llama-3-8b** (reproductible) |
| Reranker (ColBERT) | modèle | colbertv2.0 | **colbert-ir/colbertv2.0** |
| Reranker (Cross) | modèle | ms-marco-MiniLM-L-6-v2 | **cross-encoder/ms-marco-MiniLM-L-6-v2** |

## Paramètres de construction du contexte

| Variable | Valeurs testées | Protocole |
|----------|----------------|-----------|
| Format contexte | standard, citations | A, B, C |
| Ordre passages | first, middle, last | B |
| k (nombre passages) | 5, 10, 20 | A |
| Proportion distracteurs p | 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 | B |
| chunk_size | 128, 200, 256, 300, 400 | A |

## Seeds

- `RANDOM_SEED=42`
- `TORCH_SEED=42`
- `NUMPY_SEED=42`
