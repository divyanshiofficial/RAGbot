# RAGBot

**RAGBot** is a multi-document, multi-modal chatbot built using Retrieval-Augmented Generation (RAG).  
It can answer questions based on uploaded PDFs and image content, with persistent memory and semantic understanding.  
The goal is to enhance document comprehension and Q&A accuracy using embeddings and vector search.

## Features

- Upload and query multiple **PDF** and **image** files
- Uses **LangChain**, **FAISS**, and **Gemini** models (`gemini-2.0-flash`, `gemini-embedding-001`)
- Supports **semantic retrieval** for accurate, context-aware answers
- **Vector database** integration for storing and retrieving document/image embeddings
- **Streamlit UI** for an interactive, browser-based experience
- Persists user Q&A data for continuity and future analysis

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, LangChain
- **Embedding Model**: `gemini-embedding-001`
- **LLM**: Gemini (via Google Generative AI API)
- **Vector Store**: FAISS
- **File Handling**: PyMuPDF, PIL
- **Storage**: JSON for user history

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/divyanshiofficial/RAGBot.git
   cd RAGBot
