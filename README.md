# RAG Document Intelligence System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

Production-style Retrieval-Augmented Generation (RAG) system that answers questions from uploaded PDFs using semantic retrieval, reranking, and grounded LLM responses.

> Upload any PDF and ask questions in plain English.  
> Get accurate answers with exact source citations.

**Live demo:** *(coming soon — deploying to Hugging Face Spaces)*

---

# Features

- Extracts and processes PDF documents
- Creates semantically searchable vector embeddings
- Retrieves relevant chunks using FAISS similarity search
- Improves retrieval quality using Cross-Encoder reranking
- Generates grounded answers using Groq-hosted LLaMA models
- Prevents hallucinations through context-constrained prompting
- Returns source-backed responses for transparency

---

# Evaluation (RAGAS)

The system was evaluated using the RAGAS framework on a custom question-answer benchmark.

| Metric | Score |
|--------|-------|
| Faithfulness | **0.9643** |
| Answer Relevancy | **0.9986** |
| Context Precision | **0.9097** |
| Context Recall | **1.0000** |

## Interpretation

- High faithfulness indicates the model stays grounded in retrieved context with minimal hallucination.
- Near-perfect answer relevancy shows responses align strongly with user queries.
- Strong context precision demonstrates that retrieved chunks are highly useful.
- Perfect context recall indicates the retriever consistently finds all necessary information required to answer questions.

These results validate the retrieval pipeline, reranking strategy, and prompt engineering quality.

---

# Architecture

```text
PDF Upload
    │
    ▼
PyPDFLoader
    │
    ▼
RecursiveCharacterTextSplitter
(chunk_size=512, overlap=64)
    │
    ▼
HuggingFace Embeddings
(all-MiniLM-L6-v2)
    │
    ▼
FAISS Vector Store
    │
    ▼
User Question
    │
    ▼
Top-K Semantic Retrieval
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
Top-4 Most Relevant Chunks
    │
    ▼
Groq LLaMA Prompt
    │
    ▼
Answer + Source Citations
```

---

# Retrieval Pipeline

The retrieval pipeline uses a two-stage retrieval strategy:

## Stage 1 — Dense Retrieval (FAISS)

The system first retrieves semantically similar chunks using vector embeddings and FAISS similarity search.

This provides fast approximate retrieval across the document corpus.

---

## Stage 2 — Cross-Encoder Reranking

Retrieved chunks are then reranked using a Cross-Encoder model.

Unlike embedding similarity, Cross-Encoders jointly evaluate:

```text
(query, document_chunk)
```

pairs, allowing deeper semantic matching between the question and retrieved context.

This significantly improves:
- context precision
- retrieval relevance
- answer grounding

Example reranking pipeline:

```python
def rerank_docs(query, docs, top_k=4):
    pairs = [[query, doc.page_content] for doc in docs]

    scores = cross_encoder.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_k]]
```

Final context construction:

```python
def get_context(question):
    docs = retriever.invoke(question)

    reranked_docs = rerank_docs(question, docs)

    context_text = "\n\n".join(
        doc.page_content for doc in reranked_docs
    )

    context_list = [
        doc.page_content for doc in reranked_docs
    ]

    return context_text, context_list
```

---

# Tech Stack

| Component | Tool |
|-----------|------|
| PDF Parsing | PyPDFLoader (LangChain) |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Reranking | Cross-Encoder (Sentence Transformers) |
| LLM | Groq LLaMA 3.1 |
| Framework | LangChain |
| Evaluation | RAGAS |
| UI | Gradio |
| Environment | Google Colab |

---

# Key Design Decisions

## Why RecursiveCharacterTextSplitter instead of fixed chunking?

Fixed chunking blindly splits text every N characters, often breaking sentences and losing semantic meaning.

Recursive chunking attempts:
1. Paragraph splits
2. Sentence splits
3. Word-level splits

This preserves contextual integrity and improves retrieval quality significantly.

---

## Why chunk overlap?

A critical sentence can get split across chunk boundaries.

Using:

```python
chunk_overlap = 64
```

ensures important contextual information is retained across neighboring chunks, improving retrieval recall.

---

## Why reranking after retrieval?

Embedding similarity retrieval is fast but not always precise.

FAISS may retrieve chunks that are semantically related but not the most contextually relevant to the exact question.

Cross-Encoder reranking performs deeper semantic comparison between:
- user query
- retrieved chunk

This improves:
- precision
- answer quality
- grounding
- faithfulness

and reduces noisy context sent to the LLM.

---

## Why grounded prompting?

The prompt explicitly restricts the LLM to answer only from retrieved context.

This minimizes hallucinations and improves faithfulness during evaluation.

---

# How to Run

## 1. Clone repository

```bash
git clone https://github.com/your-username/rag-document-intelligence-system.git

cd rag-document-intelligence-system
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install \
langchain \
langchain-community \
sentence-transformers \
faiss-cpu \
pypdf \
gradio \
ragas \
groq
```

---

## 3. Add Groq API Key

Create a free API key from:

https://console.groq.com

Then add:

```python
GROQ_API_KEY = "your_api_key"
```

---

## 4. Run notebook

Open:

```text
RAG_document_intelligence_system.ipynb
```

in Google Colab or Jupyter Notebook and run all cells.

---

## 5. Upload PDF and ask questions

The Gradio interface will launch automatically.

---

# Example Queries

- "What are the key findings in the report?"
- "Summarize the methodology section."
- "What risks were identified?"
- "Who are the stakeholders mentioned?"
- "What conclusions does the document make?"

---

# Future Improvements

- [ ] Deploy live demo to Hugging Face Spaces
- [ ] Add conversation memory for follow-up questions
- [ ] Hybrid retrieval (FAISS + BM25)
- [ ] Multi-document querying
- [ ] Streaming responses
- [ ] Docker deployment
- [ ] Citation highlighting in PDF viewer

---

# Skills Demonstrated

This project demonstrates practical understanding of:

- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- Dense retrieval + reranking pipelines
- Embedding models
- Cross-Encoder architectures
- Prompt engineering
- LLM evaluation frameworks (RAGAS)
- LangChain pipelines
- Production-style AI system design
- End-to-end NLP workflow development

---

# About

Built as part of an AI/ML portfolio to demonstrate applied understanding of modern RAG pipelines, semantic retrieval systems, reranking architectures, and evaluation-driven LLM engineering beyond tutorial implementations.
