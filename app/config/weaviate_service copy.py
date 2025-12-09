import logging
from typing import List, Dict, Any, Optional

from weaviate.classes.config import Property, Configure, DataType
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter

from app.config.weaviate_client import VectorClient
from app.config.setting import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WeaviateService:
    """
    Weaviate v4+ vector store service.

    Features:
    - Auto schema creation
    - Hybrid / BM25 / Vector search
    - Batch upsert embeddings
    - Metadata support (user_id, chat_id, timestamps)
    """

    def __init__(self, collection_name: str = "DocumentChunk"):
        self.collection_name = collection_name

        client_instance = VectorClient()
        self.client = client_instance.client

        if not self.client or not self.client.is_ready():
            raise RuntimeError("❌ Weaviate client exists but is NOT ready.")

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
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="file_uuid", data_type=DataType.TEXT),
                    Property(name="file_name", data_type=DataType.TEXT),
                    Property(name="page_number", data_type=DataType.INT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="images", data_type=DataType.TEXT_ARRAY),

                    # Multi-tenant metadata
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="chat_id", data_type=DataType.TEXT),

                    # Optional metadata block
                    Property(
                        name="metadata",
                        data_type=DataType.OBJECT,
                        nested_properties=[
                            Property(name="created_at", data_type=DataType.TEXT),
                            Property(name="token_count", data_type=DataType.INT),
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
