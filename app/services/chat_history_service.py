from typing import List, Optional, Literal, Dict, Any
from datetime import datetime

from beanie import PydanticObjectId

from app.schemas.chat_history import ChatHistory, EmbeddedMetadata


class ChatHistoryService:

    # ------------------------
    # INSERT / SAVE
    # ------------------------
    @staticmethod
    async def add_message(
        user_id: str,
        chat_id: str,
        role: Literal["user", "assistant", "system", "tool", "supervisor", "agent"],
        content: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> ChatHistory:

        history_doc = ChatHistory(
            user_id=user_id,
            chat_id=chat_id,
            role=role,
            content=content,
            source=source,
            metadata=EmbeddedMetadata(**metadata) if metadata else EmbeddedMetadata(),
            embedding=embedding
        )

        await history_doc.insert()
        return history_doc

    # ------------------------
    # FETCH ALL MESSAGES FOR A CHAT
    # ------------------------
    @staticmethod
    async def get_chat_history(chat_id: str) -> List[ChatHistory]:
        return await ChatHistory.find(ChatHistory.chat_id == chat_id)\
            .sort("+created_at").to_list()

    # ------------------------
    # FETCH LAST N MESSAGES
    # ------------------------
    @staticmethod
    async def get_last_messages(chat_id: str, limit: int = 20) -> List[ChatHistory]:
        return await ChatHistory.find(ChatHistory.chat_id == chat_id)\
            .sort("-created_at").limit(limit).to_list()

    # ------------------------
    # FETCH BY ROLE OR AGENT
    # ------------------------
    @staticmethod
    async def filter_messages(
        chat_id: str,
        role: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[ChatHistory]:

        query = {"chat_id": chat_id}
        if role:
            query["role"] = role
        if source:
            query["source"] = source

        return await ChatHistory.find(query).sort("+created_at").to_list()

    # ------------------------
    # PAGINATION SUPPORT
    # ------------------------
    @staticmethod
    async def get_paginated(chat_id: str, skip: int = 0, limit: int = 50) -> List[ChatHistory]:
        return await ChatHistory.find(ChatHistory.chat_id == chat_id)\
            .skip(skip).limit(limit).sort("+created_at").to_list()

    # ------------------------
    # UPDATE A MESSAGE
    # ------------------------
    @staticmethod
    async def update_message(message_id: PydanticObjectId, content: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        
        update_data = {}
        if content:
            update_data["content"] = content
        if metadata:
            update_data["metadata"] = EmbeddedMetadata(**metadata)

        return await ChatHistory.find_one(ChatHistory.id == message_id).update({"$set": update_data})

    # ------------------------
    # DELETE ALL MESSAGES IN A CHAT
    # ------------------------
    @staticmethod
    async def clear_chat(chat_id: str):
        return await ChatHistory.find(ChatHistory.chat_id == chat_id).delete()

    # ------------------------
    # DELETE ONLY ASSISTANT RESPONSES
    # ------------------------
    @staticmethod
    async def delete_assistant_messages(chat_id: str):
        return await ChatHistory.find({
            "chat_id": chat_id,
            "role": "assistant"
        }).delete()

    # ------------------------
    # COUNT MESSAGES
    # ------------------------
    @staticmethod
    async def count_messages(chat_id: str) -> int:
        return await ChatHistory.find(ChatHistory.chat_id == chat_id).count()

    # ------------------------
    # GET CONTEXT FOR SUPERVISOR (TRIMMED)
    # ------------------------
    @staticmethod
    async def get_context_window(chat_id: str, max_tokens: int = 3000):
        """
        Returns a shortened conversation history so supervisor agent 
        stays inside token budget.

        (Optional) In future you can integrate summarizer here.
        """
        messages = await ChatHistoryService.get_last_messages(chat_id, 40)

        # Basic heuristic token control — replace later with tokenizer call
        token_estimate = 0
        final_msgs = []

        for msg in reversed(messages):
            token_estimate += len(msg.content.split())
            if token_estimate > max_tokens:
                break
            final_msgs.append(msg)

        return list(reversed(final_msgs))
