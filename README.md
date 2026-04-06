# rag-doc-intelligence
 AI-powered document Q&amp;A system using RAG, LangChain &amp; FAISS
 # RAG Document Intelligence System

> Upload any PDF and ask questions in plain English. 
> Get accurate answers with exact source citations.

**Live demo:** [link when deployed on HF Spaces]

## What this does
- Uploads and processes any PDF document
- Splits it intelligently into searchable chunks
- Uses semantic search to find relevant sections
- GPT-3.5 answers questions using ONLY the document

## Evaluation results
| Metric | Score | What it means |
|--------|-------|---------------|
| Faithfulness | 0.XX | Answers stayed true to document |
| Answer Relevancy | 0.XX | Answers matched the questions |
| Context Precision | 0.XX | Right chunks were retrieved |
| Context Recall | 0.XX | Found all relevant information |

## Tech stack
LangChain · sentence-transformers · FAISS · OpenAI GPT-3.5 · Gradio

## How to run
1. Open the notebook in Google Colab
2. Add your OpenAI API key to Colab Secrets
3. Run all cells in order
4. Upload a PDF and start asking questions
