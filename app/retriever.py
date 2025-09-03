import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever():
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # If FAISS index exists, load it; else build it
    if os.path.exists("vector_store"):
        db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    else:
        raise RuntimeError("Vector store not found. Run indexer.py first to build it.")

    return db.as_retriever()
