import os
import tempfile
import fitz  # PyMuPDF
import uuid
import logging
import asyncio
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.utils.chunker import get_text_splitter
from app.services.s3_service import S3Service
from app.config.weaviate_service import WeaviateService

# Mongo Models
from app.schemas.uploaded_file import UploadedFile, FileStatus
from app.schemas.chunk_record import ChunkRecord
from app.schemas.chat_session import ChatSession

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class FileServiceBatch:
    """
    Full RAG pipeline:
    - Save file to S3
    - Extract text, chunk, vectorize
    - Save chunks to Weaviate
    - Track all documents & chunks in MongoDB
    """

    def __init__(
        self,
        s3_service: S3Service,
        weaviate_service: Optional[WeaviateService] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.s3 = s3_service
        self.weaviate = weaviate_service or WeaviateService()
        self.device = _get_device()

        logger.info(f"🧠 Embedding Model Device: {self.device}")

        self.embed_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)

        logger.info(f"🔧 Models Loaded: {embedding_model_name}, {reranker_model_name}")

    # ---------------------------------------------------------------
    # Batch Processing
    # ---------------------------------------------------------------
    async def process_files(self, files: List[Any], filenames: List[str], metadata: Dict[str, str]) -> Dict[str, Any]:
        results = {}

        user_id = metadata.get("user_id")
        chat_id = metadata.get("chat_id")

        chat_session = await ChatSession.find_one(ChatSession.chat_id == chat_id)
        if not chat_session:
            chat_session = ChatSession(user_id=user_id, chat_id=chat_id)
            await chat_session.insert()

        for file, filename in zip(files, filenames):
            results[filename] = await self._process_single_file(file, filename, user_id, chat_id, chat_session)

        return results

    # ---------------------------------------------------------------
    # Single file pipeline
    # ---------------------------------------------------------------
    async def _process_single_file(
        self, file: Any, filename: str, user_id: str, chat_id: str, chat_document: ChatSession
    ) -> Dict[str, Any]:

        file_bytes = await file.read()
        file_uuid = str(uuid.uuid4())

        # 1️⃣ Upload original file to S3
        s3_key = f"files/{file_uuid}/{filename}"
        file_url = self.s3.upload_bytes(file_bytes, key=s3_key, content_type=file.content_type)

        # Initial record before processing
        stored_file = UploadedFile(
            user_id=user_id,
            chat=chat_document,
            file_id=file_uuid,
            file_name=filename,
            size_bytes=len(file_bytes),
            mime_type=file.content_type,
            s3_url=file_url,               # Public HTTP URL of file in S3
            images=[],                    # ADD: placeholder list for extracted images
            status=FileStatus.processing,
        )

        await stored_file.insert()


        # Temp save for parsing
        _, ext = os.path.splitext(filename)
        suffix = ext or ".bin"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            elements = partition(filename=tmp_path, include_page_breaks=True, extract_images_in_pdf=False)

            # 2️⃣ Extract images (PDF only)
            page_images = {}
            if suffix.lower() == ".pdf":
                pdf = fitz.open(tmp_path)
                for page_num in range(len(pdf)):
                    uploaded = []
                    for img_index, img_info in enumerate(pdf[page_num].get_images(full=True)):
                        xref = img_info[0]
                        img = pdf.extract_image(xref)
                        img_bytes = img["image"]
                        ext = img.get("ext", "png")
                        img_key = f"files/{file_uuid}/page-{page_num+1}/img-{img_index}.{ext}"

                        url = self.s3.upload_bytes(img_bytes, img_key, content_type=f"image/{ext}")
                        uploaded.append(url)

                    page_images[page_num + 1] = uploaded

            # 3️⃣ Chunking
            splitter = get_text_splitter()
            final_chunks = []
            section_title = None
            logical_page = 1

            for el in elements:
                raw_text = (getattr(el, "text", "") or "").strip()
                meta = getattr(el, "metadata", None)
                page_num = getattr(meta, "page_number", None) or logical_page

                if getattr(el, "category", None) == "PageBreak":
                    logical_page += 1
                    continue

                if getattr(el, "category", None) in ["Title", "Header"]:
                    section_title = raw_text

                parts = splitter.split_text(raw_text) if raw_text else [""]

                for part in parts:
                    final_chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": part,
                        "file_url": file_url,
                        "page_number": page_num,
                        "parent_section": section_title,
                        "file_uuid": file_uuid,
                        "images": page_images.get(page_num, []),
                        "metadata": {"file_name": filename},
                    })

            # 4️⃣ Embeddings
            vectors = await asyncio.get_event_loop().run_in_executor(
                _MODEL_EXECUTOR, lambda: self.embed_model.encode([c["text"] for c in final_chunks], convert_to_numpy=True)
            )

            weaviate_objects = []

            # 5️⃣ Save chunks
            for index, (chunk, emb) in enumerate(zip(final_chunks, vectors)):
                properties = {
                    "text": chunk["text"],
                    "file_uuid": chunk["file_uuid"],
                    "file_name": filename,
                    "file_url": file_url,
                    "page_number": chunk["page_number"],
                    "chunk_index": index,
                    "images": chunk["images"],
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "metadata": {
                        "token_count": len(chunk["text"].split()),
                        "created_at": stored_file.created_at.isoformat(),
                        "document_url": file_url,
                    },
                }

                weaviate_objects.append({
                    "id": chunk["id"],
                    "vector": emb.tolist(),
                    "properties": properties,
                })

                # await ChunkRecord(
                #     user_id=user_id,
                #     chat_id=chat_id,
                #     file=stored_file,
                #     weaviate_id=chunk["id"],
                #     page_number=chunk["page_number"],
                #     chunk_index=index,
                #     token_count=len(chunk["text"].split()),
                #     preview_text=chunk["text"][:200],
                #     images=chunk["images"],
                #     file_url=file_url,
                #     metadata=properties["metadata"],
                # ).insert()

            # Update file summary
            stored_file.total_chunks = len(final_chunks)
            stored_file.page_count = len(page_images)
            stored_file.status = FileStatus.completed
            await stored_file.save()

            await self.weaviate.upsert_objects(weaviate_objects)
            

            all_images = list({img for arr in page_images.values() for img in arr})

            stored_file.add_images(all_images)  

            stored_file.total_chunks = len(final_chunks)
            stored_file.page_count = len(page_images)

            stored_file.update_progress(FileStatus.completed)

            await stored_file.save()


            return {
                "file_uuid": file_uuid,
                "chunks": len(final_chunks),
                "file_url": file_url,
                "images": list({img for arr in page_images.values() for img in arr}),
            }

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ---------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------
    async def query(self,user_id:str,chat_id:str, text: str, top_k: int = 10, rerank_top: int = 20,):
        query_vec = await asyncio.get_event_loop().run_in_executor(
            _MODEL_EXECUTOR, lambda: self.embed_model.encode([text], convert_to_numpy=True)[0]
        )

        results = self.weaviate.hybrid_search(
            user_id=user_id,
            chat_id=chat_id,
            query=text,
            embedding=query_vec.tolist(),
            alpha=0.5,
            top_k=rerank_top
        )

        if not results:
            return []

        rerank_scores = self.reranker.predict([[text, r["properties"].get("text", "")] for r in results])

        for r, s in zip(results, rerank_scores):
            r["rerank_score"] = float(s)
            r["final_score"] = (0.6 * (r.get("score") or 0)) + (0.4 * float(s))

        return sorted(results, key=lambda x: x["final_score"], reverse=True)[:top_k]

    @staticmethod
    def _normalize_type(raw: str) -> str:
        return {"Title": "Title", "Header": "Header", "Image": "Image", "NarrativeText": "NarrativeText"}.get(raw, "Text")
