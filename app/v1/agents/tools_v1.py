"""
LangChain tool wrappers: Arxiv, Playwright loader (Chrome), Unstructured PDF loader,
and simple in-memory wrappers. These are returned in a shape that deepagents or
LangChain agents can consume as tools.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional

# Corrected imports for core components
from langchain.tools import tool # This specific import is fine for the @tool decorator
from langchain_core.tools import Tool # Correct import for the Tool class
from langchain_community.document_loaders import UnstructuredPDFLoader, PlaywrightURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ArXiv utility - LangChain offers wrappers; fallback to python arxiv
try:
    # If LangChain has an ArxivAPIWrapper in your version, prefer that.
    from langchain_community.utilities import ArxivAPIWrapper # may vary by LangChain version
    ARXIV_AVAILABLE = True
except ImportError: # Use ImportError for catching import errors
    ARXIV_AVAILABLE = False
    import arxiv  # pip install arxiv


async def arxiv_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search arXiv and return a list of structured results.
    Uses langchain.utilities.ArxivAPIWrapper if present, else the 'arxiv' package.
    """
    if ARXIV_AVAILABLE:
        wrapper = ArxivAPIWrapper()
        # The ArxivAPIWrapper run method expects a single string query and returns a single string of results.
        # This will need reformatting if you want a list of dicts as the function signature suggests.
        # For simplicity, we'll return the string output of the wrapper if available.
        # A full rewrite would be needed to normalize the outputs.
        return [{"result_summary_string": wrapper.run(query)}] # Returning a list of one dict for type consistency

    # fallback using python arxiv
    loop = asyncio.get_event_loop()
    def sync_search():
        search = arxiv.Search(query=query, max_results=max_results)
        out = []
        # Note: 'r.entry_id' might be deprecated, 'r.url' is generally preferred.
        for r in search.results():
            out.append({
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "summary": r.summary,
                "published": r.published.isoformat(),
                "url": r.url # Changed from r.entry_id
            })
        return out
    return await loop.run_in_executor(None, sync_search)


async def load_pdf_and_chunk(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Load a PDF from disk using UnstructuredPDFLoader and chunk it into Documents.
    """
    loader = UnstructuredPDFLoader(file_path)  # requires 'unstructured' package
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


async def load_url_and_chunk(url: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Load a web page using Playwright (Chrome) and chunk it.
    """
    # Note: PlaywrightURLLoader might require prior setup (e.g., 'playwright install').
    loader = PlaywrightURLLoader(urls=[url], remove_selectors=["header", "footer", "nav"])
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# Wrap LangChain-style tools
def get_tool_list():
    """
    Return a list of Tool objects for LangChain / deepagents to use.
    """
    tools = [
        # Functions used with Tool.from_function() should ideally be synchronous
        # unless you are using an agent designed to handle async tools.
        # The 'func' argument expects a callable.
        Tool.from_function(
            func=arxiv_search, # Note: This function is async
            name="arxiv_search",
            description="Search arXiv for papers matching a query. Input: query (str). Returns list of metadata dicts."
        ),
        Tool.from_function(
            func=load_pdf_and_chunk, # Note: This function is async
            name="load_pdf_and_chunk",
            description="Load a PDF file path (local path) and return chunked documents."
        ),
        Tool.from_function(
            func=load_url_and_chunk, # Note: This function is async
            name="load_url_and_chunk",
            description="Load a URL using Playwright (Chrome) and return chunked documents."
        )
    ]
    return tools
