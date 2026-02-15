import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


st.title("📄 AI Document Search Engine")

DATA_PATH = "data"
DB_PATH = "vector_db"

# -------------------- Upload PDFs --------------------
uploaded_files = st.file_uploader(
    "Upload PDF files", accept_multiple_files=True, type="pdf"
)

if uploaded_files:
    os.makedirs(DATA_PATH, exist_ok=True)

    for file in uploaded_files:
        with open(f"{DATA_PATH}/{file.name}", "wb") as f:
            f.write(file.read())

    st.success("Files uploaded successfully")


# -------------------- Index Documents --------------------
if st.button("Index Documents"):
    if not os.path.exists(DATA_PATH):
        st.warning("Upload PDFs first")
        st.stop()

    documents = []

    for file in os.listdir(DATA_PATH):
        loader = PyPDFLoader(f"{DATA_PATH}/{file}")
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

    st.success("Documents indexed successfully")


# -------------------- Ask Questions --------------------
query = st.text_input("Ask a question about your documents")

if query:

    # stop if user didn’t index first
    if not os.path.exists(DB_PATH):
        st.warning("⚠️ Please click 'Index Documents' first")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # local open-source LLM
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_length=256
    )

    docs = db.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])

    # better prompt = cleaner answers
    prompt = f"""
You are a helpful academic assistant.

Read the context and answer the question in SIMPLE words.
Do NOT copy formulas unless needed.
Give a short answer in 3-4 sentences.

Context:
{context}

Question: {query}

Answer:
"""


    result = llm(prompt)
    answer = result[0]["generated_text"].replace(prompt, "").strip()

    st.markdown("### 🤖 Answer")
    st.write(answer)

    st.markdown("### 📚 Sources")
    for d in docs:
        st.write(f"Page {d.metadata.get('page')} - {d.metadata.get('source')}")
