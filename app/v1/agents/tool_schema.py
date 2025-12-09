TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": "Search arXiv for papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Search the web using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "High quality web research using Tavily.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_url_and_chunk",
            "description": "Load URL and return text chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "chunk_size": {"type": "integer"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_pdf_and_chunk",
            "description": "Load a PDF and chunk text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "chunk_size": {"type": "integer"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_recall",
            "description": "Retrieve relevant memory from vector store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "chat_id": {"type": "string"},
                    "query": {"type": "string"},
                    "k": {"type": "integer"}
                },
                "required": ["user_id", "chat_id", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_memory",
            "description": "Save text into memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "chat_id": {"type": "string"},
                    "text": {"type": "string"}
                },
                "required": ["user_id", "chat_id", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "citation_formatter",
            "description": "Format citations in APA/IEEE style.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "object"}},
                    "style": {"type": "string"}
                },
                "required": ["items", "style"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "outline_generator",
            "description": "Generate a research paper outline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_sections": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }
]
