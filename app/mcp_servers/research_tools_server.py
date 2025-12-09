import logging
import asyncio
import json
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

# =========================================================
# LOGGING SETUP
# =========================================================

logger = logging.getLogger("mcp.research_tools")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

ResearchTools = FastMCP("ResearchTools")


# =========================================================
# HELPERS
# =========================================================

async def run_sync(func, *args):
    """Run sync blocking functions in thread pool."""
    logger.debug(f"[run_sync] Running sync function: {func.__name__}, args={args}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(None, lambda: func(*args))
    logger.debug(f"[run_sync] Completed {func.__name__}")
    return result


def _log_payload(name: str, payload: Dict[str, Any]):
    """Uniform logging of payloads with safe truncation."""
    payload_str = json.dumps(payload, ensure_ascii=False)
    safe = payload_str[:4000]
    if len(payload_str) > 4000:
        safe += "...[truncated]"
    logger.info(f"[{name}] Payload => {safe}")


# =========================================================
# SEARCH TOOLS
# =========================================================

@ResearchTools.tool()
async def arxiv_search(query: str, max_results: int = 5) -> str:
    logger.info(f"[arxiv_search] called: query='{query}', max_results={max_results}")

    from app.agents.tools import arxiv_search as _arxiv_search
    results = await run_sync(_arxiv_search, query, max_results)

    payload = {
        "source": "arxiv",
        "query": query,
        "count": len(results) if isinstance(results, list) else 1,
        "results": results,
    }
    _log_payload("arxiv_search", payload)
    return json.dumps(payload, ensure_ascii=False)


@ResearchTools.tool()
async def duckduckgo_search(query: str, max_results: int = 5) -> str:
    logger.info(f"[duckduckgo_search] called: query='{query}', max_results={max_results}")

    from app.agents.tools import duckduckgo_search as _duckduckgo_search
    results = await run_sync(_duckduckgo_search, query, max_results)

    payload = {
        "source": "duckduckgo",
        "query": query,
        "count": len(results) if isinstance(results, list) else 1,
        "results": results,
    }
    _log_payload("duckduckgo_search", payload)
    return json.dumps(payload, ensure_ascii=False)


@ResearchTools.tool()
async def pubmed_search(query: str, max_results: int = 5) -> str:
    logger.info(f"[pubmed_search] called: query='{query}', max_results={max_results}")

    from app.agents.tools import pubmed_search as _pubmed_search
    results = await run_sync(_pubmed_search, query, max_results)

    payload = {
        "source": "pubmed",
        "query": query,
        "count": len(results) if isinstance(results, list) else 1,
        "results": results,
    }
    _log_payload("pubmed_search", payload)
    return json.dumps(payload, ensure_ascii=False)


@ResearchTools.tool()
async def semantic_scholar_search(query: str, max_results: int = 5) -> str:
    logger.info(f"[semantic_scholar_search] called: query='{query}', max_results={max_results}")

    from app.agents.tools import semantic_scholar_search as _semantic_scholar_search
    results = await run_sync(_semantic_scholar_search, query, max_results)

    payload = {
        "source": "semantic_scholar",
        "query": query,
        "count": len(results) if isinstance(results, list) else 1,
        "results": results,
    }
    _log_payload("semantic_scholar_search", payload)
    return json.dumps(payload, ensure_ascii=False)


# =========================================================
# CITATION TOOLS
# =========================================================

@ResearchTools.tool()
async def citation_extract(text: str) -> str:
    logger.info(f"[citation_extract] called: text_length={len(text or '')}")

    from app.agents.tools import citation_extraction
    result = await run_sync(citation_extraction, text)

    payload = {
        "operation": "citation_extract",
        "input_length": len(text or ""),
        "result": result,
    }
    _log_payload("citation_extract", payload)
    return json.dumps(payload, ensure_ascii=False)


@ResearchTools.tool()
async def citation_format(items: Any, style: str = "APA") -> str:
    logger.info(f"[citation_format] called: style={style}, raw_items_type={type(items)}")

    from app.agents.tools import citation_formatter

    try:
        # Attempt JSON parsing if string
        if isinstance(items, str):
            try:
                items = json.loads(items)
                logger.debug("[citation_format] JSON string parsed successfully")
            except Exception:
                logger.warning("[citation_format] Failed to parse items as JSON")
                return json.dumps({
                    "error": "citation_format: invalid JSON string",
                    "items_raw": items
                }, ensure_ascii=False)

        if items is None:
            items = []

        if isinstance(items, dict):
            items = [items]

        if not isinstance(items, list):
            return json.dumps({
                "error": "citation_format: expected list",
                "received_type": str(type(items))
            }, ensure_ascii=False)

        normalized_items = []
        for it in items:
            if isinstance(it, dict):
                normalized_items.append(it)
            else:
                try:
                    normalized_items.append(json.loads(it))
                except Exception:
                    normalized_items.append({"title": str(it)})

        result = await run_sync(citation_formatter, normalized_items, style)

        payload = {
            "operation": "citation_format",
            "style": style,
            "count": len(normalized_items),
            "result": result,
        }
        _log_payload("citation_format", payload)
        return json.dumps(payload, ensure_ascii=False)

    except Exception as exc:
        logger.exception("[citation_format] Exception")
        import traceback
        return json.dumps({
            "error": "citation_format: exception",
            "exception": str(exc),
            "traceback": traceback.format_exc()[:2000],
        }, ensure_ascii=False)


# =========================================================
# OTHER TOOLS
# =========================================================

@ResearchTools.tool()
async def crossref_search(query: str, max_results: int = 5) -> str:
    logger.info(f"[crossref_search] called: query='{query}', max_results={max_results}")

    from app.agents.tools import crossref_search
    results = await run_sync(crossref_search, query, max_results)

    payload = {"source": "crossref", "query": query, "results": results}
    _log_payload("crossref_search", payload)
    return json.dumps(payload, ensure_ascii=False)


@ResearchTools.tool()
async def load_url(url: str, chunk_size: int = 1000) -> str:
    logger.info(f"[load_url] called: url='{url}', chunk_size={chunk_size}")

    from app.agents.tools import load_url_and_chunk
    chunks = await run_sync(load_url_and_chunk, url, chunk_size)

    payload = {"url": url, "chunk_size": chunk_size, "chunks": chunks}
    _log_payload("load_url", payload)
    return json.dumps(payload, ensure_ascii=False)


@ResearchTools.tool()
async def load_pdf(file_path: str, chunk_size: int = 1000) -> str:
    logger.info(f"[load_pdf] called: file_path='{file_path}', chunk_size={chunk_size}")

    from app.agents.tools import load_pdf_and_chunk
    chunks = await run_sync(load_pdf_and_chunk, file_path, chunk_size)

    payload = {"file_path": file_path, "chunk_size": chunk_size, "chunks": chunks}
    _log_payload("load_pdf", payload)
    return json.dumps(payload, ensure_ascii=False)


@ResearchTools.tool()
async def save_memory(user_id: str, chat_id: str, text: str) -> str:
    logger.info(f"[save_memory] called: user_id={user_id}, chat_id={chat_id}, text_len={len(text or '')}")

    from app.agents.tools import save_to_memory
    result = await run_sync(save_to_memory, user_id, chat_id, text)

    payload = {"operation": "save_memory", "user_id": user_id, "chat_id": chat_id, "result": result}
    _log_payload("save_memory", payload)
    return json.dumps(payload, ensure_ascii=False)


@ResearchTools.tool()
async def recall_memory(user_id: str, chat_id: str, query: str, k: int = 4) -> str:
    logger.info(f"[recall_memory] called: user_id={user_id}, chat_id={chat_id}, k={k}")

    from app.agents.tools import memory_recall
    result = await run_sync(memory_recall, user_id, chat_id, query, k)

    payload = {"operation": "recall_memory", "user_id": user_id, "chat_id": chat_id, "query": query, "result": result}
    _log_payload("recall_memory", payload)
    return json.dumps(payload, ensure_ascii=False)


@ResearchTools.tool()
async def outline_generator(query: str, max_sections: int = 6) -> str:
    logger.info(f"[outline_generator] called: query='{query}', max_sections={max_sections}")

    from app.agents.tools import outline_generator
    result = await run_sync(outline_generator, query, max_sections)

    payload = {"operation": "outline_generator", "query": query, "result": result}
    _log_payload("outline_generator", payload)
    return json.dumps(payload, ensure_ascii=False)


@ResearchTools.tool()
async def export_pdf(draft: Dict[str, Any], out_path: str = None) -> str:
    logger.info(f"[export_pdf] called: draft_keys={list(draft.keys())}, out_path={out_path}")

    from app.agents.tools import export_pdf
    result = await run_sync(export_pdf, draft, out_path)

    payload = {"operation": "export_pdf", "out_path": out_path, "result": result}
    _log_payload("export_pdf", payload)
    return json.dumps(payload, ensure_ascii=False)


# =========================================================
# SERVER ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    logger.info("🚀 MCP ResearchTools server starting at http://0.0.0.0:8008/mcp")

    ResearchTools.settings.port = 8008
    ResearchTools.settings.host = "0.0.0.0"

    logger.info("[Server] Starting FastMCP server…")
    ResearchTools.run(transport="streamable-http")
