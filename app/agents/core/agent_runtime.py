import logging
import numpy as np
from datetime import datetime

from app.services.chat_history_service import ChatHistoryService
from app.config.weaviate_service import WeaviateService
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger("AgentRuntime")


class AgentRuntime:
    """
    A universal wrapper that adds:
    - Vector-based memory recall
    - Message summarization when history gets long
    - Automatic forgetting (relevance-based pruning)
    - Vector memory graph linking (semantic association saving)
    - Context injection before calling any agent
    
    Works with any agent that supports `.run()` and `.stream()`.
    """

    def __init__(self, agent, llm):
        self.agent = agent
        self.llm = llm

        self.history = ChatHistoryService()
        self.embedding = EmbeddingService()
        self.weaviate = WeaviateService()

        # config knobs
        self.summary_threshold = 40
        self.forget_threshold_count = 200
        self.relevance_threshold = 0.35

    # ----------------------------------------------------------------
    # 🧠 MEMORY PIPELINE
    # ----------------------------------------------------------------
    async def _retrieve_context(self, user_id: str, chat_id: str, query: str):
        """
        Retrieve:
         - Last 5 conversation messages
         - Top vector memories
         - Knowledge graph linked memories
        """

        recent_msgs = await self.history.get_last_messages(chat_id, limit=5)
        formatted_chat = "\n".join([f"**{m.role.upper()}**: {m.content}" for m in recent_msgs])

        vector_results = await self.weaviate.query(user_id, chat_id, query, top_k=5)
        formatted_vector = "\n".join([f"- {r['properties'].get('text','')}" for r in vector_results])

        # GRAPH MEMORY — store and retrieve connections between ideas
        await self._link_memory_to_graph(user_id, chat_id, query)
        graph_related = await self.weaviate.query(user_id, chat_id, query, top_k=3)

        formatted_graph = "\n".join([f"- {r['properties'].get('text','')}" for r in graph_related])

        return f"""
### Recent Chat:
{formatted_chat}

### Retrieved Relevant Knowledge:
{formatted_vector}

### Linked Knowledge Graph Context:
{formatted_graph}

### New User Query:
{query}
"""

    # ----------------------------------------------------------------
    # 🔗 MEMORY GRAPH — stores semantic relationships
    # ----------------------------------------------------------------
    async def _link_memory_to_graph(self, user_id, chat_id, text):
        """
        Creates vector similarity edges by storing triple nodes:
        (previous_memory) -> (query)
        """
        embedding = await self.embedding.embed(text)
        await self.weaviate.save_chat_turn(
            user_id=user_id,
            chat_id=chat_id,
            text=text,
            embedding=embedding,
            role="graph-link",
            agent="memory-kernel",
            metadata={"link_type": "semantic_edge"}
        )

    # ----------------------------------------------------------------
    # 🧪 RELEVANCE SCORING
    # ----------------------------------------------------------------
    async def _score_relevance(self, message_text: str, context: str) -> float:
        query_emb = await self.embedding.embed(context)
        text_emb = await self.embedding.embed(message_text)

        similarity = float(np.dot(text_emb, query_emb) /
                           (np.linalg.norm(text_emb) * np.linalg.norm(query_emb)))

        prompt = f"""
Rate the memory below on long-term usefulness for future reasoning.
Return ONLY 0, 0.5 or 1.

Memory: "{message_text}"
"""
        result = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        importance_score = float(result.content.strip()) if result.content.strip().replace(".", "").isdigit() else 0.0

        return (similarity * 0.45) + (importance_score * 0.10)

    # ----------------------------------------------------------------
    # ✂️ SUMMARIZATION
    # ----------------------------------------------------------------
    async def _summarize_old_messages(self, user_id: str, chat_id: str):
        messages = await self.history.get_chat_history(chat_id)
        if len(messages) < self.summary_threshold:
            return

        logger.warning("⚠ Summarizing older conversation history...")

        older_msgs = messages[:-10]
        convo_text = "\n".join([f"{m.role.upper()}: {m.content}" for m in older_msgs])

        prompt = f"""
Summarize this conversation into a permanent memory block capturing:
- Key facts
- Preferences
- Decisions
- Confirmed knowledge
- Open questions

Conversation:
{convo_text}
"""
        summary_response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
        summary_text = summary_response.content

        await self._store_memory(user_id, chat_id, "system", summary_text, "conversation-summary")

        for msg in older_msgs:
            await self.history.update_message(msg.id, content="[Summarized]")

    # ----------------------------------------------------------------
    # 🗑️ FORGETTING
    # ----------------------------------------------------------------
    async def _forget_irrelevant(self, user_id: str, chat_id: str, query: str):
        messages = await self.history.get_chat_history(chat_id)
        if len(messages) < self.forget_threshold_count:
            return

        logger.warning(f"🧹 Forgetting irrelevant memories ({len(messages)} stored)...")

        scored = []
        for m in messages:
            score = await self._score_relevance(m.content, query)
            scored.append({"id": m.id, "text": m.content, "score": score})

        scored.sort(key=lambda x: x["score"])

        to_delete = [m for m in scored if m["score"] < self.relevance_threshold]

        for item in to_delete:
            await self.history.update_message(item["id"], content="[Forgotten — low relevance]")

    # ----------------------------------------------------------------
    # 🏦 PERSIST MEMORY
    # ----------------------------------------------------------------
    async def _store_memory(self, user_id, chat_id, role, text, agent_label):
        emb = await self.embedding.embed(text)

        await self.history.add_message(
            user_id=user_id,
            chat_id=chat_id,
            role=role,
            content=text,
            source=agent_label,
            embedding=emb,
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )

        await self.weaviate.save_chat_turn(
            user_id=user_id,
            chat_id=chat_id,
            text=text,
            embedding=emb,
            role=role,
            agent=agent_label
        )

    # ----------------------------------------------------------------
    # 🚀 EXECUTION ENTRY POINT
    # ----------------------------------------------------------------
    async def run(self, user_id: str, chat_id: str, query: str):
        await self._store_memory(user_id, chat_id, "user", query, "input")
        await self._summarize_old_messages(user_id, chat_id)
        await self._forget_irrelevant(user_id, chat_id, query)

        context = await self._retrieve_context(user_id, chat_id, query)
        result = await self.agent.run(user_id, chat_id, context)

        await self._store_memory(user_id, chat_id, "assistant", result["response"], "agent-output")
        return result

    async def stream(self, user_id: str, chat_id: str, query: str):
        await self._store_memory(user_id, chat_id, "user", query, "input")
        await self._summarize_old_messages(user_id, chat_id)
        await self._forget_irrelevant(user_id, chat_id, query)

        context = await self._retrieve_context(user_id, chat_id, query)

        async for token in self.agent.stream(user_id, chat_id, context):
            yield token

        await self._store_memory(user_id, chat_id, "assistant", "[STREAMED RESPONSE]", "agent-output")
