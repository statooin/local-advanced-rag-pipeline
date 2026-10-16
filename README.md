# Local RAG Pipeline: Hybrid Retrieval & Reranking 🔍🧠

**Production-ready RAG pipeline on local LLMs. Built for enterprise knowledge bases.**

[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain%200.3-blue.svg)](https://github.com/langchain-ai/langchain)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant%201.12-red.svg)](https://qdrant.tech/)
[![BGE-M3](https://img.shields.io/badge/Embeddings-BGE--M3-orange.svg)](https://huggingface.co/BAAI/bge-m3)
[![Infrastructure](https://img.shields.io/badge/Deploy-Docker%20%7C%20Compose-2496ED.svg)](#)

> ⚙️ WORK IN PROGRESS: Active development. Core pipeline stable; evaluation harness in progress.

---

## 🎯 The Enterprise RAG Challenge

Naive RAG — single vector search, basic chunking, no reranking — fails in production on dense technical documentation: it misses exact terminology, loses context across chunk boundaries, and delivers low-precision results that erode user trust.

This pipeline solves this with a **three-stage retrieval architecture**:

1. **Chunk:** Parent-Child chunking with Contextual Summaries prefix — preserves semantic context across boundaries.
2. **Retrieve:** BGE-M3 hybrid search (dense + sparse → RRF fusion in Qdrant) — captures both semantic similarity and exact term matching in a single forward pass.
3. **Rerank:** BGE-Reranker-v2-m3 cross-encoder — joint attention over full parent context delivers BEIR nDCG@10 = 51.8 precision.

---

## 🏗️ Architecture

```
Documents (.pdf / .txt)
        │
        ▼
 ┌─────────────┐
 │  Docling    │  PDF → Markdown (CPU, cached)
 └──────┬──────┘
        │
        ▼
 ┌─────────────────────┐
 │  Contextual Summary │  LLM (Ollama / LLM-agnostic API) → doc-level prefix
 └──────────┬──────────┘
            │
            ▼
 ┌──────────────────────────────┐
 │  Parent-Child Chunking       │  Parent ~512 tok → LLM context
 │  + Contextual Prefix         │  Child  ~200 tok → embedded in Qdrant
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │  BGE-M3 Hybrid Indexing      │  Dense (1024-dim) + Sparse (lexical)
 │  Qdrant Local DB             │  Single forward pass per batch
 └──────────────┬───────────────┘
                │
          Query Time
                │
                ▼
 ┌──────────────────────────────┐
 │  Stage 1: Hybrid Search      │  Dense + Sparse → RRF → top-20
 │  (Qdrant native Query API)   │  No manual score fusion
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │  Stage 2: Cross-Encoder      │  BGE-Reranker-v2-m3
 │  Reranking                   │  top-20 → top-5 by joint attention
 └──────────────┬───────────────┘
                │
                ▼
 ┌──────────────────────────────┐
 │  Stage 3: LCEL Refine Chain  │  Initial answer → iterative refinement
 │  (Streaming)                 │  per reranked parent context
 └──────────────────────────────┘
```

---

## 🔬 Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **BGE-M3 dense + sparse in one pass** | Eliminates the overhead of running two separate models; sparse catches exact technical terms that dense embeddings miss |
| **Parent-Child chunking** | Child chunks (200 tok) keep embedding precision high; parent chunks (512 tok) give the LLM full context for generation |
| **Contextual Summary prefix** | Anthropic (2024): reduces retrieval misses by ~67% by anchoring each chunk to its document-level meaning |
| **RRF fusion on Qdrant side** | No manual weight tuning; reciprocal rank fusion is robust across query types without hyperparameter search |
| **Cross-encoder reranker** | Bi-encoders (used in Stage 1) approximate relevance; cross-encoder joint attention is 2× more precise at acceptable latency |
| **SHA-256 cache invalidation** | Docling parse cache, summary cache, and chunk cache each carry a content+params hash — stale on any change, instant on repeat |

---

## ⚙️ Stack

| Layer | Technology |
|---|---|
| **LLM** | Ollama (local) / LLM-agnostic API via LangChain |
| **Embeddings** | `BAAI/bge-m3` — FlagEmbedding, dense + sparse |
| **Reranker** | `BAAI/bge-reranker-v2-m3` — CrossEncoder |
| **Vector DB** | Qdrant (local file mode, HNSW M=16) |
| **PDF Parsing** | Docling (CPU, PyPdfium2 backend, table recovery) |
| **Orchestration** | LangChain LCEL — Refine Chain with streaming |
| **Infra** | Docker Compose, JupyterLab, named volumes |

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 16 GB RAM (CPU) or 8 GB VRAM (GPU)
- ~8 GB disk for model cache (BGE-M3 + Reranker + LLM)

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/local-rag-pipeline-astra-linux.git
cd local-rag-pipeline-astra-linux

cp .env.example .env
# Fill in LLM_API_KEY
# or leave empty to use local Ollama (qwen3.5:9b pulled automatically)
nano .env
```

### 2. Add Documents

```bash
# Drop any .pdf or .txt files into:
cp your_docs/*.pdf data/docs/

# Or pull the example docs:
curl -o data/docs/doc1.pdf https://your-source/doc.pdf
```

### 3. Build & Run

```bash
# CPU (default):
docker compose up --build

# GPU (uncomment deploy block in docker-compose.yml first):
# docker compose up --build
```

JupyterLab: `http://localhost:8888` → token: `masterclass`

### 4. Run the Notebook

Execute cells in order:

```
Cell 0    → LLM connection (Ollama or YandexGPT)
Cell 1a/b → Document loading & Docling PDF parsing (cached)
Cell 2a/b → Text cleaning & Contextual Summaries (LLM, cached)
Cell 2.5  → BGE-M3 initialization (downloaded once to volume)
Cell 3a/b → Parent-Child chunking (SHA-256 cached)
Cell 4    → BGE-M3 dense + sparse demo
Cell 5    → Qdrant collection setup (idempotent)
Cell 6a/b → Hybrid indexing (idempotent, upsert)
Cell 7a/b → Hybrid Search demo (Dense + Sparse → RRF)
Cell 8a/b → Reranker demo (cross-encoder)
Cell 9a/b → Full pipeline query with streaming Refine Chain
```

---

## 📁 Project Structure

```
local-rag-pipeline/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── entrypoint.sh
├── .env                ← cp .env.example .env
├── .env.example
├── .gitignore
├── notebooks/
│   ├── notebook_03_rag_simple.ipynb   ← ChromaDB baseline RAG
│   └── notebook_04_rag_base.ipynb     ← Full hybrid pipeline
└── data/
    └── docs/                          ← Drop .pdf / .txt here
```

---

## 🗺️ Roadmap

- [ ] **Evaluation harness:** RAGAS metrics (faithfulness, answer relevancy, context precision) on a labeled Q&A set
- [ ] **Query rewriting:** HyDE (Hypothetical Document Embeddings) for improved sparse retrieval on abstract queries
- [ ] **Multi-turn memory:** Conversation buffer with sliding context window for follow-up questions
- [ ] **REST API:** FastAPI wrapper over the pipeline for integration with external applications
- [ ] **GPU benchmarks:** Latency/throughput comparison table CPU vs. CUDA across indexing and query stages

---

## 👨‍💻 Author

**Ilya Averyanov**
*Senior Site Reliability Engineer | AI Infrastructure & Platform Engineering*

Specializing in production-grade AI systems, cloud-native infrastructure, and high-performance data pipelines.
