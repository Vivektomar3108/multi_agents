import os
import asyncio
import logging
from dotenv import load_dotenv
from pprint import pprint

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

from app.services.pdf_service import FileServiceBatch
from app.config.aws_s3 import S3Service
from app.config.weaviate_service import WeaviateService

# ---------------------------
# ENV + LOGGING
# ---------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ELEN-Agent")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ Missing GROQ_API_KEY in environment variables!")

# ---------------------------
# LLM initialization
# ---------------------------
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.1,
)

# ---------------------------
# TOOL: weaviate-based retrieval (robust + closes connections)
# ---------------------------
@tool
async def weaviate_index_search(query: str) -> str:
    """
    Search the indexed documents in Weaviate using embedding + hybrid + reranking,
    return a formatted string combining top chunks.

    This function ensures any client it creates is closed before returning to avoid
    unclosed transport warnings.
    """
    s3 = None
    weaviate = None
    file_service = None
    try:
        # create fresh clients per tool call (so we can close them)
        s3 = S3Service()
        weaviate = WeaviateService()
        file_service = FileServiceBatch(s3, weaviate_service=weaviate)

        # tune top_k if you need more context
        results = await file_service.query(text=query, top_k=8)

        if not results:
            return "⚠️ No matching indexed knowledge found."

        # Results might be dict-like or object-like. Be defensive.
        formatted_chunks = []
        for i, r in enumerate(results):
            # support both dict and object
            text = None
            if isinstance(r, dict):
                text = r.get("properties", {}).get("text") or r.get("text") or str(r)
            else:
                # attempt to read attributes gracefully
                props = getattr(r, "properties", None) or getattr(r, "to_dict", lambda: {})()
                if isinstance(props, dict):
                    text = props.get("text") or props.get("content") or str(r)
                else:
                    text = str(r)
            # limit chunk length to avoid excessively long tool output if desired
            if isinstance(text, str) and len(text) > 4000:
                text = text[:3900] + "\n\n... (truncated)"
            formatted_chunks.append(f"📄 **Chunk {i+1}:**\n{text}")

        combined = "\n\n---\n".join(formatted_chunks)
        # return both a short summary header and the combined text for agent consumption
        return f"FOUND {len(formatted_chunks)} CHUNKS\n\n{combined}"

    except Exception as e:
        logger.exception("Error in weaviate_index_search")
        return f"ERROR: {e}"

    finally:
        # best-effort cleanup: close weaviate client if present (may be sync or async)
        try:
            if weaviate is not None and hasattr(weaviate, "client"):
                close_fn = getattr(weaviate.client, "close", None) or getattr(weaviate.client, "aclose", None)
                if close_fn:
                    maybe = close_fn()
                    if asyncio.iscoroutine(maybe):
                        await maybe
        except Exception:
            logger.exception("Failed to close weaviate client in tool cleanup")


TOOLS = [weaviate_index_search]

# ---------------------------
# SYSTEM PROMPT (for the agent)
# ---------------------------
SYSTEM_PROMPT = """
You are ELEN — a precise research assistant.

RULES:
- Use ONLY the information returned from the tool `weaviate_index_search`.
- No hallucination. No filler text. No generic statements.
- If information is missing, state it briefly at the end (“Note: Some details are not available in the retrieved data.”).

OUTPUT FORMAT (clean Markdown):
weaviate_index_search
## Summary
- 1–3 crisp bullet points summarizing what the retrieved sources contain.

## Answer
A clear, concise, professional explanation directly answering the user’s question.  
Use short paragraphs or bullet points. Be factual and to the point.

## Note (only if needed)
One short line describing missing or incomplete data.


"""
# ---------------------------
# Build the agent (LangChain v1 style)
# ---------------------------
agent = create_agent(
    model=llm,               # use `model=` argument
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)

# ---------------------------
# Debug-friendly run function
# ---------------------------
async def run_agent(question: str, debug: bool = True):
    print(f"\n🔍 USER QUESTION: {question}\n")

    try:
        inputs = {
            "messages": [
                {"role": "user", "content": question},
            ],
            # you can pass other run-time options here if needed
        }

        # Async invoke on the agent
        result = await agent.ainvoke(inputs)

        # DEBUG: print full raw result so you can inspect structure
        if debug:
            print("\n--- RAW AGENT RESULT (pprint) ---")
            pprint(result)
            print("--- END RAW RESULT ---\n")

        # Extract the model/assistant final message
        messages = result.get("messages", [])
        answer = None
        if messages:
            last = messages[-1]
            # many LangChain BaseMessage objects have .content
            answer = getattr(last, "content", None) or str(last)
        else:
            # older/alternate APIs: some results put final text in "output" or "text"
            answer = result.get("output") or result.get("text") or "⚠️ No answer generated"

        print("\n==============================")
        print("🤖 FINAL ANSWER")
        print("==============================\n")
        print(answer)

    except Exception as e:
        logger.exception("Agent execution failed")
        print(f"❌ Agent execution failed: {e}")

    finally:
        # nothing to close here for the agent itself; ensure any global resources are closed if you have them
        print("Run complete. If you still see unclosed transport warnings, check that all clients created inside tools are closed.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    QUERY = "Explain compliance requirements for crowdfunding under India's Section 8 law."
    asyncio.run(run_agent(QUERY, debug=True))
