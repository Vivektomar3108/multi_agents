# app/agents/tools.py
import os
import re
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

import requests
import feedparser
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

from app.config.chroma import get_vector_store
from app.schemas.memory import MemoryEntry

logger = logging.getLogger(__name__)

# Optional Tavily import
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False

TAVILY_KEY = os.getenv("TAVILY_API_KEY")


# ====================================================================
# --------------------------- SEARCH TOOLS ----------------------------
# ====================================================================

# ------------------------- arXiv Search ------------------------------
def arxiv_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    try:
        logger.info("[arXiv] query=%s", query)
        q = quote_plus(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={int(max_results)}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        results = []
        for e in feed.entries[:max_results]:
            authors = [a.name for a in getattr(e, "authors", [])]
            results.append({
                "title": getattr(e, "title", ""),
                "authors": authors,
                "summary": getattr(e, "summary", ""),
                "published": getattr(e, "published", ""),
                "url": getattr(e, "link", ""),
                "source": "arxiv",
            })
        return results
    except Exception as e:
        logger.exception("[arXiv] failed: %s", e)
        return []


# ---------------------- DuckDuckGo Search ---------------------------
def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    try:
        logger.info("[DDG] Searching: %s", query)
        url = "https://api.duckduckgo.com"
        params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()

        data = r.json()
        results = []
        for t in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(t, dict) and t.get("Text"):
                results.append({
                    "title": t["Text"],
                    "url": t.get("FirstURL"),
                    "snippet": t["Text"],
                    "source": "duckduckgo"
                })
        return results
    except Exception as e:
        logger.exception("[DDG] failed: %s", e)
        return []


# --------------------------- Tavily Search ---------------------------
def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    if not TAVILY_AVAILABLE or not TAVILY_KEY:
        logger.warning("[TAVILY] Not available.")
        return []

    try:
        logger.info("[TAVILY] query=%s", query)
        client = TavilyClient(api_key=TAVILY_KEY)
        res = client.search(query, max_results=max_results, include_raw_content=True)
        return res.get("results", res)
    except Exception as e:
        logger.exception("[TAVILY] failed: %s", e)
        return []


# --------------------------- PubMed Search ---------------------------
def pubmed_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    NCBI E-Utilities pubmed search.
    """
    try:
        logger.info("[PubMed] query=%s", query)
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base}esearch.fcgi?db=pubmed&retmode=json&term={quote_plus(query)}&retmax={max_results}"
        search_res = requests.get(search_url, timeout=20).json()
        ids = search_res.get("esearchresult", {}).get("idlist", [])

        if not ids:
            return []

        # Fetch metadata
        fetch_url = f"{base}efetch.fcgi?db=pubmed&id={','.join(ids)}&retmode=xml"
        xml_res = requests.get(fetch_url, timeout=20).text

        soup = BeautifulSoup(xml_res, "xml")
        out = []
        for article in soup.find_all("PubmedArticle"):
            title = article.find("ArticleTitle")
            abstract = article.find("AbstractText")
            year = article.find("PubDate")
            out.append({
                "title": title.text if title else "",
                "summary": abstract.text if abstract else "",
                "published": year.text if year else "",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.PMID.text}/",
                "authors": [],
                "source": "pubmed"
            })
        return out[:max_results]
    except Exception as e:
        logger.exception("[PubMed] failed: %s", e)
        return []


# ---------------------- Semantic Scholar Search ----------------------
def semantic_scholar_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Free Semantic Scholar API (no key needed for basic search)
    """
    try:
        logger.info("[S2] query=%s", query)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote_plus(query)}&limit={max_results}&fields=title,year,abstract,authors,url"
        res = requests.get(url, timeout=30).json()
        out = []
        for p in res.get("data", []):
            out.append({
                "title": p.get("title"),
                "summary": p.get("abstract"),
                "published": str(p.get("year")),
                "authors": [a["name"] for a in p.get("authors", [])],
                "url": p.get("url"),
                "source": "semantic_scholar"
            })
        return out
    except Exception as e:
        logger.exception("[S2] failed: %s", e)
        return []


# -------------------------- CrossRef Search ---------------------------
def crossref_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    try:
        logger.info("[CrossRef] query=%s", query)
        url = f"https://api.crossref.org/works?query={quote_plus(query)}&rows={max_results}"
        data = requests.get(url, timeout=20).json()
        items = data.get("message", {}).get("items", [])
        out = []
        for i in items:
            out.append({
                "title": i.get("title", [""])[0],
                "authors": [a.get("family", "") + ", " + a.get("given", "") for a in i.get("author", [])],
                "summary": i.get("abstract", ""),
                "published": str(i.get("created", {}).get("date-parts", [[None]])[0][0]),
                "url": i.get("URL"),
                "source": "crossref",
                "doi": i.get("DOI")
            })
        return out
    except Exception as e:
        logger.exception("[CrossRef] failed: %s", e)
        return []


# ====================================================================
# --------------------------- LOADERS --------------------------------
# ====================================================================
def load_url_and_chunk(url: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
    try:
        logger.info("[Loader] URL=%s", url)
        r = requests.get(url, timeout=30)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script", "style", "noscript"]): s.decompose()
        text = soup.get_text("\n")

        chunks = []
        i = 0
        while i < len(text):
            chunks.append({
                "page_content": text[i:i + chunk_size],
                "meta": {"source": url}
            })
            i += chunk_size

        logger.debug("[Loader] URL chunks=%s", len(chunks))
        return chunks
    except Exception as e:
        logger.exception("[Loader] load_url failed: %s", e)
        return []


def load_pdf_and_chunk(file_path: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """
    PDF loader using PyPDF2 fallback.
    """
    try:
        logger.info("[Loader] PDF=%s", file_path)
        import PyPDF2
        reader = PyPDF2.PdfReader(file_path)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])

        chunks = []
        i = 0
        while i < len(text):
            chunks.append({
                "page_content": text[i:i + chunk_size],
                "meta": {"source": file_path}
            })
            i += chunk_size

        logger.debug("[Loader] PDF chunks=%s", len(chunks))
        return chunks
    except Exception as e:
        logger.exception("[Loader] load_pdf failed: %s", e)
        return []


# ====================================================================
# ---------------------------- MEMORY --------------------------------
# ====================================================================
def save_to_memory(user_id: str, chat_id: str, text: str) -> Dict[str, Any]:
    try:
        vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
        vs.add_texts([text])
        asyncio.run(_store_short_window(user_id, chat_id, "system", text))
        return {"status": "ok"}
    except Exception as e:
        logger.exception("[Memory] save failed: %s", e)
        return {"status": "error", "error": str(e)}


async def _store_short_window(user_id: str, chat_id: str, role: str, text: str):
    try:
        entry = await MemoryEntry.find_one(
            MemoryEntry.user_id == user_id,
            MemoryEntry.chat_id == chat_id,
            MemoryEntry.key == "short_window_messages"
        )
        if entry:
            arr = entry.value.get("messages", [])
            arr.append({"role": role, "text": text})
            arr = arr[-8:]
            entry.value = {"messages": arr}
            await entry.save()
        else:
            await MemoryEntry(
                user_id=user_id,
                chat_id=chat_id,
                key="short_window_messages",
                value={"messages":[{"role": role, "text": text}]}
            ).insert()
    except Exception as e:
        logger.exception("[Memory] window update failed: %s", e)


def memory_recall(user_id: str, chat_id: str, query: str, k: int = 4) -> List[Dict[str, Any]]:
    try:
        vs = get_vector_store(collection=f"user_{user_id}_chat_{chat_id}")
        docs = vs.similarity_search(query, k=k)
        return [{"page_content": d.page_content, "meta": d.metadata} for d in docs]
    except Exception as e:
        logger.exception("[Memory] recall failed: %s", e)
        return []


# ====================================================================
# --------------------- CITATION EXTRACTION ---------------------------
# ====================================================================
def citation_extraction(text: str) -> List[Dict[str, Any]]:
    """
    Regex-based lightweight citation extraction.
    Example matches: "Smith et al., 2020", "Johnson & Lee (2018)", "Doe, J. (2021)"
    """
    try:
        logger.info("[CitationExtract] extracting citations")

        pattern = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+et al\.| & [A-Z][a-z]+)?\s*\(?(\d{4})\)?"
        matches = re.findall(pattern, text)

        results = []
        for m in matches:
            results.append({
                "authors": [m[0]],
                "year": m[1],
                "title": "",
                "url": "",
                "source": "extracted"
            })
        return results
    except Exception as e:
        logger.exception("[CitationExtract] failed: %s", e)
        return []


# ====================================================================
# -------------------------- CITATION FORMAT --------------------------
# ====================================================================
def citation_formatter(items: List[Dict[str, Any]], style: str = "APA") -> Dict[str, Any]:
    """
    Robust citation formatter:
    - Accepts items where authors may be under 'authors' (list) or 'author' (string).
    - Accepts 'published' or 'year' or 'published_date' fields.
    - Ensures output is a JSON-serializable dict with 'style' and 'references' keys.
    - Catches exceptions and returns an error structure.
    """
    try:
        refs = []

        for p in items or []:
            # ensure p is a dict
            if not isinstance(p, dict):
                # coerce
                p = {"title": str(p)}

            # Normalize authors into a list of strings
            authors = []
            if "authors" in p and isinstance(p["authors"], list):
                # if elements are dicts, try to extract name
                for a in p["authors"]:
                    if isinstance(a, dict):
                        # common keys: name, given/family
                        name = a.get("name") or " ".join(filter(None, [a.get("given", ""), a.get("family", "")])).strip()
                        if not name:
                            name = str(a)
                        authors.append(name)
                    else:
                        authors.append(str(a))
            elif "author" in p:
                # single author string
                if isinstance(p["author"], list):
                    authors = [str(x) for x in p["author"]]
                else:
                    authors = [str(p["author"])]
            else:
                # fallback: maybe 'creator' or empty
                if "creator" in p:
                    if isinstance(p["creator"], list):
                        authors = [str(x) for x in p["creator"]]
                    else:
                        authors = [str(p["creator"])]
                else:
                    authors = []

            # Year extraction: prefer 'year' then parse 'published' or 'published_date'
            year = None
            if p.get("year"):
                year = str(p.get("year"))
            else:
                pub = p.get("published") or p.get("published_date") or p.get("date") or ""
                # try to extract 4-digit year
                m = None
                if isinstance(pub, str):
                    import re
                    m = re.search(r"(19|20)\d{2}", pub)
                    if m:
                        year = m.group(0)
                # fallback to empty string
                year = year or ""

            title = p.get("title") or p.get("name") or ""
            url = p.get("url") or p.get("link") or ""

            # Build reference text based on style
            if style and style.upper() == "APA":
                a = ", ".join(authors) if authors else ""
                # safe formatting
                if year:
                    refs.append(f"{a} ({year}). {title}. {url}".strip())
                else:
                    refs.append(f"{a}. {title}. {url}".strip())

            elif style and style.upper() == "IEEE":
                a = ", ".join(authors) if authors else ""
                # If year missing, use placeholder
                y = year if year else "n.d."
                refs.append(f"[{y}] {a}. {title}. Available at: {url}".strip())
            else:
                # default simple representation
                refs.append(f"{title} — {', '.join(authors)} — {year} — {url}".strip())

        # Ensure all refs are plain strings
        refs = [str(r) for r in refs]

        return {"style": style, "references": refs}
    except Exception as e:
        logger.exception("[CitationFormatter] failed: %s", e)
        return {"style": style, "references": [], "error": str(e)}



# ====================================================================
# --------------------------- OUTLINE --------------------------------
# ====================================================================
def outline_generator(query: str, max_sections: int = 6) -> Dict[str, Any]:
    try:
        base_sections = [
            "Abstract",
            "Introduction",
            "Background",
            "Methodology",
            "Experiments",
            "Results",
            "Discussion",
            "Conclusion"
        ]
        sections = [{"name": s, "text": ""} for s in base_sections[:max_sections]]
        return {"outline": sections}
    except Exception as e:
        logger.exception("[Outline] generation failed: %s", e)
        return {"outline": []}


# ====================================================================
# --------------------------- PDF EXPORT -------------------------------
# ====================================================================
def export_pdf(draft: Dict[str, Any], out_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Saves the final research paper JSON as a readable PDF.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        if out_path is None:
            out_path = "research_output.pdf"

        c = canvas.Canvas(out_path, pagesize=A4)
        text_obj = c.beginText(40, 800)
        text_obj.setFont("Helvetica", 10)

        content = json.dumps(draft, indent=2, ensure_ascii=False)
        for line in content.split("\n"):
            if len(line) > 120:
                chunks = [line[i:i+120] for i in range(0, len(line), 120)]
                for ch in chunks:
                    text_obj.textLine(ch)
            else:
                text_obj.textLine(line)

        c.drawText(text_obj)
        c.save()

        logger.info("[PDF] exported to %s", out_path)
        return {"out_path": out_path}
    except Exception as e:
        logger.exception("[PDF] export failed: %s", e)
        return {"error": str(e)}
