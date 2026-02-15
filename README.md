# AI Document Search Engine (RAG App)

An end-to-end Retrieval Augmented Generation (RAG) application that lets users upload PDF documents and ask questions about them using a local LLM.

This project demonstrates how modern AI assistants are built using embeddings, vector databases, and large language models.

---

## Features

• Upload multiple PDF documents  
• Automatic document chunking & embedding generation  
• Semantic search using FAISS vector database  
• Question answering using a local HuggingFace LLM  
• Source page citation for answers  
• Interactive Streamlit web interface  

---

## Tech Stack

Python  
Streamlit  
LangChain  
FAISS Vector Database  
HuggingFace Transformers  
SentenceTransformers  

---

## How It Works

1. User uploads PDF documents  
2. Documents are split into chunks  
3. Each chunk is converted into embeddings  
4. Embeddings are stored in FAISS vector DB  
5. User asks a question  
6. System retrieves relevant chunks  
7. LLM generates an answer using retrieved context  

This architecture is known as **RAG (Retrieval Augmented Generation)**.

---

## Run Locally

Clone the repository:
```bash
git clone https://github.com/Vithesh-4/ai-document-search-engine.git
cd ai-document-search-engine

# create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux

# install dependencies
pip install -r requirements.txt

# run app
streamlit run app.py
