import logging
from typing import List, Dict, Any, Optional

from weaviate.classes.config import Property, Configure, DataType
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter

from app.config.weaviate_client import VectorClient
from app.services.embedding_service import EmbeddingService
from app.config.setting import settings
from sentence_transformers import  CrossEncoder


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class WeaviateService:
    """
    Weaviate v4+ vector store service.

    Features:
    - Auto schema creation
    - Hybrid / BM25 / Vector search
    - Batch upsert embeddings
    - Metadata support (user_id, chat_id, timestamps)
    - Stores AI + User conversation memory per chat session (generic across multiple agents)
    """

    def __init__(self, collection_name: str = "DocumentChunk",reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",):
        self.collection_name = collection_name

        client_instance = VectorClient()
        self.client = client_instance.client

        if not self.client or not self.client.is_ready():
            raise RuntimeError("❌ Weaviate client exists but is NOT ready.")
        


        # Embedding service & optional reranker
        self.embedding_service = EmbeddingService()
        self.device = _get_device()
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)

        logger.info("✅ Weaviate connected and ready.")
        self._ensure_schema()

    # ------------------------------------------------------------
    # Schema Definition
    # ------------------------------------------------------------
    def _ensure_schema(self) -> None:
        """Create collection if missing."""
        try:
            existing = self.client.collections.list_all()
        except Exception as e:
            logger.exception(f"❌ Failed to fetch schema: {e}")
            raise

        if self.collection_name in existing:
            logger.info(f"📦 Collection '{self.collection_name}' already exists.")
            return

        logger.info(f"🛠 Creating collection schema: '{self.collection_name}'")

        try:
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),   # manual embeddings
                properties=[
                    # Core text storage for ANY agent / chat turn
                    Property(name="text", data_type=DataType.TEXT),

                    # NEW — Generic conversational memory fields
                    Property(name="role", data_type=DataType.TEXT),      # "user" or "assistant"
                    Property(name="agent", data_type=DataType.TEXT),     # research_agent / writer_agent / system
                    Property(name="source_type", data_type=DataType.TEXT),  # chat / tool_result / file / summary

                    # Legacy file-based schema (kept untouched)
                    Property(name="file_uuid", data_type=DataType.TEXT),
                    Property(name="file_name", data_type=DataType.TEXT),
                    Property(name="page_number", data_type=DataType.INT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="images", data_type=DataType.TEXT_ARRAY),

                    # Multi-tenant metadata
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="chat_id", data_type=DataType.TEXT),

                    # Optional metadata block — expanded for agent memory
                    Property(
                        name="metadata",
                        data_type=DataType.OBJECT,
                        nested_properties=[
                            Property(name="created_at", data_type=DataType.TEXT),
                            Property(name="token_count", data_type=DataType.INT),
                            Property(name="context_type", data_type=DataType.TEXT),  # query, response, memory, summary
                            Property(name="embedding_model", data_type=DataType.TEXT),
                        ]
                    ),
                ],
            )

            logger.info(f"🎉 Collection '{self.collection_name}' created successfully.")

        except Exception as e:
            logger.exception(f"❌ Failed to create schema: {e}")
            raise

    # ------------------------------------------------------------
    # UPSERT (Batch Insert)
    # ------------------------------------------------------------
    async def upsert_objects(
        self, objects: List[Dict[str, Any]], batch_size: int = 50
    ) -> None:

        collection = self.client.collections.get(self.collection_name)
        total = len(objects)
        success_count = 0

        logger.info(f"📦 Inserting {total} chunks in batches of {batch_size}...")

        for i in range(0, total, batch_size):
            batch = objects[i:i + batch_size]
            payloads: List[DataObject] = []

            for obj in batch:
                try:
                    payloads.append(
                        DataObject(
                            uuid=obj.get("id"),
                            properties=obj.get("properties") or {},
                            vector=obj.get("vector"),  # only here, not in properties
                        )
                    )
                except Exception as e:
                    logger.exception(f"⚠️ Failed formatting object {obj.get('id')}")

            if not payloads:
                continue

            try:
                collection.data.insert_many(payloads)
                success_count += len(payloads)
            except Exception as e:
                logger.exception(f"❌ Batch insert error at index {i}: {e}")

        logger.info(f"✅ Insert complete: {success_count}/{total} stored.")

    # ------------------------------------------------------------
    # NEW — Save chat turn memory
    # ------------------------------------------------------------
    async def save_chat_turn(
        self,
        user_id: str,
        chat_id: str,
        text: str,
        embedding: List[float],
        role: str,
        agent: str,
        metadata: Optional[dict] = None
    ):
        """
        Save user or assistant messages in Weaviate as vector memory.
        Works for ANY agent.
        """

        obj = {
            "id": None,
            "vector": embedding,
            "properties": {
                "text": text,
                "role": role,
                "agent": agent,
                "source_type": "chat",
                "user_id": user_id,
                "chat_id": chat_id,
                "metadata": metadata or {},
            }
        }

        return await self.upsert_objects([obj])

    # ------------------------------------------------------------
    # 🧠 Lexical Search (BM25)
    # ------------------------------------------------------------
    def bm25_search(
        self,
        query: str,
        user_id: str,
        chat_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:

        collection = self.client.collections.get(self.collection_name)

        weaviate_filter = Filter.by_property("user_id").equal(user_id)

        if chat_id:
            weaviate_filter = weaviate_filter & Filter.by_property("chat_id").equal(chat_id)

        result = collection.query.bm25(
            query=query,
            limit=top_k,
            filters=weaviate_filter,
            return_metadata=MetadataQuery(score=True),
        )

        return self._format_results(result.objects, score_field="score")

    # ------------------------------------------------------------
    # 🔍 Vector Search
    # ------------------------------------------------------------
    def vector_search(
        self,
        embedding: List[float],
        user_id: str,
        chat_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:

        collection = self.client.collections.get(self.collection_name)

        weaviate_filter = Filter.by_property("user_id").equal(user_id)

        if chat_id:
            weaviate_filter = weaviate_filter & Filter.by_property("chat_id").equal(chat_id)

        result = collection.query.near_vector(
            near_vector=embedding,
            limit=top_k,
            filters=weaviate_filter,
            return_metadata=MetadataQuery(distance=True),
        )

        return self._format_results(result.objects, score_field="distance")

    # ------------------------------------------------------------
    # 🧪 Hybrid Search
    # ------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        embedding: Optional[List[float]],
        user_id: str,
        chat_id: Optional[str] = None,
        alpha: float = 0.5,
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:

        collection = self.client.collections.get(self.collection_name)

        weaviate_filter = Filter.by_property("user_id").equal(user_id)

        if chat_id:
            weaviate_filter = weaviate_filter & Filter.by_property("chat_id").equal(chat_id)

        result = collection.query.hybrid(
            query=query,
            alpha=alpha,
            vector=embedding,
            limit=top_k,
            filters=weaviate_filter,
            return_metadata=MetadataQuery(score=True),
        )

        return self._format_results(result.objects, score_field="score")

    # ------------------------------------------------------------
    # Utility formatter
    # ------------------------------------------------------------
    @staticmethod
    def _format_results(objects, score_field: str) -> List[Dict[str, Any]]:
        formatted = []

        for obj in objects:
            metadata = getattr(obj, "metadata", None)
            score = getattr(metadata, score_field) if metadata and hasattr(metadata, score_field) else None

            formatted.append(
                {
                    "id": getattr(obj, "uuid", None),
                    "score": score,
                    "properties": getattr(obj, "properties", {}),
                    "metadata": metadata,
                }
            )

        return formatted



    # ------------------------------------------------------------
    # QUERY (Corrected)
    # ------------------------------------------------------------
    async def query(self, user_id: str, chat_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Smart search: hybrid (text + vector similarity). Optional reranking if enabled.
        """

        # 1️⃣ Embed query
        query_vec = await self.embedding_service.embed(query)

        # 2️⃣ Filter per user + chat
        weaviate_filter = Filter.by_property("user_id").equal(user_id)
        if chat_id:
            weaviate_filter = weaviate_filter & Filter.by_property("chat_id").equal(chat_id)

        # 3️⃣ Perform hybrid search
        collection = self.client.collections.get(self.collection_name)

        result = collection.query.hybrid(
            query=query,
            alpha=0.5,
            vector=query_vec,
            limit=top_k,
            filters=weaviate_filter,
            return_metadata=MetadataQuery(score=True),
        )

        results = self._format_results(result.objects, score_field="score")

        # 4️⃣ Optional rerank stage
        if self.reranker:
            rerank_pairs = [[query, r["properties"].get("text", "")] for r in results]
            scores = self.reranker.predict(rerank_pairs)

            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
                r["final_score"] = (0.6 * r["score"]) + (0.4 * float(s))

            return sorted(results, key=lambda x: x["final_score"], reverse=True)

        return results