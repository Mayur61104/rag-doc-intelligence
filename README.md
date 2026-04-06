# RAG Document Intelligence System

> Upload any PDF and ask questions in plain English.  
> Get accurate answers with exact source citations.

**Live demo:** *(coming soon — deploying to Hugging Face Spaces)*

---

## What this does

- Uploads and processes any PDF document
- Splits it intelligently into searchable chunks using recursive character splitting
- Uses semantic search (FAISS) to find the most relevant sections
- Groq LLaMA answers questions using **only** the document — no hallucination
- Returns the exact source chunks used to generate every answer

---

## Evaluation results

Since RAGAS requires OpenAI (paid), I built a **custom 3-layer evaluation pipeline** instead — which actually gives more transparency into where the system succeeds and fails.

### Method 1 — Manual exact match (4 questions)
Ground truth answers written by hand, compared against system output.

| Result | Score |
|--------|-------|
| Correct answers | 4 / 4 |
| Accuracy | **100%** |

### Method 2 — LLM-as-judge (10 questions)
Groq LLaMA scores each answer 1–10 for correctness, relevance and faithfulness.

| Result | Score |
|--------|-------|
| LLM validation score | **8.5 / 10** |

### Method 3 — Semantic similarity (cosine)
Embeddings of system answers vs ground truth answers compared using cosine similarity.

| Result | Score |
|--------|-------|
| Average cosine similarity | **0.64** |
| Interpretation | Strong semantic overlap — answers capture the right meaning even when wording differs |

> **Why custom evaluation?** Standard RAG evaluation frameworks (RAGAS) depend on OpenAI's paid API. Building a multi-method evaluation pipeline from scratch demonstrates a deeper understanding of how to measure model quality — which is exactly what production data science roles require.

---

## Architecture

```
PDF Upload
    │
    ▼
PyPDFLoader → RecursiveCharacterTextSplitter (chunk_size=512, overlap=64)
    │
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2) → FAISS Vector Store
    │
    ▼
User Question → Embed → Retrieve top-4 chunks → Groq LLaMA prompt
    │
    ▼
Answer + Source citations
```

---

## Tech stack

| Component | Tool |
|-----------|------|
| PDF parsing | PyPDFLoader (LangChain) |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS (Facebook AI) |
| LLM | Groq LLaMA (free tier) |
| Evaluation | Custom — cosine similarity + LLM-as-judge + manual |
| UI | Gradio |
| Environment | Google Colab |

---

## Key decisions & what I learned

**Why recursive chunking over fixed-size?**  
Fixed chunking splits blindly every N characters, often mid-sentence. Recursive splitting tries paragraph breaks first, then sentences, then words — keeping meaning intact. This directly improves retrieval quality.

**Why chunk_overlap=64?**  
Without overlap, a key fact split across two chunk boundaries is never fully retrieved. 64 characters of shared context ensures no sentence is ever completely lost.

**Why build custom evaluation instead of RAGAS?**  
RAGAS requires a paid OpenAI key. Rather than blocking on cost, I designed a 3-layer evaluation: manual ground truth for precision, LLM-as-judge for scalability, and cosine similarity for semantic correctness. This multi-method approach is more robust than a single automated score.

---

## How to run

1. Open `RAG_document_intelligence_system.ipynb` in [Google Colab](https://colab.research.google.com)
2. Add your Groq API key to Colab Secrets as `GROQ_API_KEY`  
   *(Free at console.groq.com — no credit card needed)*
3. Run all cells in order (top to bottom)
4. Upload any PDF using the file picker
5. Ask questions in the Gradio interface

---

## What's next

- [ ] Deploy live demo to Hugging Face Spaces
- [ ] Add conversation memory (follow-up questions)
- [ ] Hybrid search: combine FAISS semantic search with BM25 keyword search
- [ ] Multi-document support (query across multiple PDFs simultaneously)
- [ ] Add `requirements.txt`

---

## About

Built as part of a data science portfolio project to demonstrate end-to-end RAG system design, custom evaluation methodology, and production-thinking beyond standard tutorials.
