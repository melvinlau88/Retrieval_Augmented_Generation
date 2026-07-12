# Movie RAG

A small RAG (Retrieval-Augmented Generation) project that answers questions about movies, built by working through LangChain's [RAG From Scratch](https://github.com/langchain-ai/rag-from-scratch) tutorial and swapping in live TMDb movie data instead of the tutorial's blog post example.

## What it does

Fetches popular movies from the TMDb API, indexes them in a local Chroma vector database, and answers questions like *"popular action movies right now"* or *"highly rated movies from the 2010s"* using retrieval-augmented generation.

## Tech stack

- LangChain
- Groq (free LLM — `llama-3.3-70b-versatile`)
- HuggingFace Sentence Transformers (free local embeddings)
- Chroma (local vector store)
- TMDb API

## Setup

1. Install dependencies:
```bash
   pip install langchain langchain-community langchain-huggingface langchain-groq langchain-text-splitters chromadb sentence-transformers python-dotenv requests --break-system-packages
```

2. Create a `.env` file:
GROQ_API_KEY=your_key_here
TMDB_API_TOKEN=your_token_here

3. Run:
```bash
   python main.py
```

## Features

- Multi-query retrieval (rewords questions 5 ways for better search coverage)
- Logical + semantic routing
- Structured query filtering (e.g. rating, release date)
