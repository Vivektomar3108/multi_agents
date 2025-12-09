# app/config/memory.py
from app.config.chroma import get_vector_store
from app.schemas.memory import MemoryEntry

def get_vector(collection_name: str):
    return get_vector_store(collection=collection_name)
