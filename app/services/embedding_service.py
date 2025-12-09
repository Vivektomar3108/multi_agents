import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from typing import List, Union

# Thread executor so encode doesn't block event loop
_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=2)

class EmbeddingService:
    """
    Handles async embeddings using SentenceTransformer, optimized to not block event loop.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load embedding model once
        self.embed_model = SentenceTransformer(embedding_model_name, device=self.device)

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings async.
        Accepts single string or list of strings.
        """

        if isinstance(text, str):
            text = [text]

        loop = asyncio.get_event_loop()

        vectors = await loop.run_in_executor(
            _MODEL_EXECUTOR,
            lambda: self.embed_model.encode(text, convert_to_numpy=True).tolist()
        )

        return vectors[0] if len(vectors) == 1 else vectors
