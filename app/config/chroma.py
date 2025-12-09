import os
from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_memory")

client = PersistentClient(path=CHROMA_DIR)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store(collection: str = "memory"):
    return Chroma(
        client=client,
        collection_name=collection,
        embedding_function=embedding_model,
    )
