import json
import logging
from typing import Dict, Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient


logger = logging.getLogger("SearchAgent")


class SearchAgent:
    """
    Web search agent that integrates with AgentRuntime for automatic memory management.
    Handles both memory-based queries and fresh web searches.
    """

    def __init__(self, llm, max_results: int = 7):
        self.llm = llm
        self.max_results = max_results
        self.mcp_client = None
        self.tools = []
        self.agent = None

    async def initialize(self):
        """Initialize MCP client and configure agent."""
        self.tools = await self._configure_tools()
        
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self._system_prompt(),
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,
                    trigger=('tokens', 4000),
                    keep=('messages', 15)
                )
            ]
        )
        return self

    async def _configure_tools(self):
        """Configure MCP search tools for the agent."""
        self.mcp_client = MultiServerMCPClient(
            {
                "web_search": {
                    "transport": "stdio",
                    "command": "uvx",
                    "args": [
                        "--from", "git+https://github.com/pranavms13/web-search-mcp",
                        "web-search-mcp"
                    ],
                }
            }
        )
        
        tools = await self.mcp_client.get_tools()
        return tools

    def _system_prompt(self):
        return """
You are a Web Search Agent with full conversation memory awareness.

**CONTEXT AWARENESS:**
You receive enriched context from the memory system:
- Recent Chat: Previous conversation messages
- Retrieved Relevant Knowledge: Related stored information
- Linked Knowledge Graph Context: Semantic associations
- New User Query: The current question

**QUERY HANDLING:**

1. **Memory Questions** (answer from context):
   - "what did we discuss", "do you remember", "earlier", "before"
   - Follow-up questions referencing conversation
   → Answer using PROVIDED CONTEXT ONLY (no web search)
   → Reference what was previously discussed

2. **New Information Questions** (use web search):
   - Current events, news, reviews, comparisons
   - Facts not in conversation history
   - "How to", tutorials, latest information
   - Questions about people, places, events
   - ANY question where you need to verify or get fresh information
   → ALWAYS use search tools to fetch fresh data
   → Search for 3-5 different sources to get comprehensive information
   → Even if you think you know the answer, use search to verify
   → Verify information from multiple reliable sources

**SEARCH STRATEGY:**
- Use max_results between 3-7 to get multiple perspectives
- Prefer diverse, authoritative sources
- Cross-verify facts across multiple results

**IMPORTANT:** When in doubt, ALWAYS search. It's better to search and verify than to rely on potentially outdated information.

**RESPONSE RULES:**
- Be concise and specific (2-4 sentences)
- Include key details: numbers, dates, names
- For web searches: cite reliable sources
- For memory queries: acknowledge conversation context
- Only state verified facts

**OUTPUT (Markdown format):**
Provide a clear, factual answer that directly addresses the question.
For web searches, include source URLs.
For memory queries, reference the conversation naturally.
"""

    async def run(self, user_id: str, chat_id: str, query: str) -> Dict[str, Any]:
        """
        Execute search with full context awareness.
        Query contains enriched context from AgentRuntime with chat history and knowledge.
        """
        logger.info(f"[SearchAgent] Processing query for user={user_id}, chat={chat_id}")
        
        try:
            # Invoke agent with full context (includes chat history and retrieved knowledge)
            response = await self.agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            
            # Debug: Log the response structure
            logger.info(f"[SearchAgent] Response type: {type(response)}")
            logger.info(f"[SearchAgent] Response keys: {response.keys() if isinstance(response, dict) else 'N/A'}")
            if isinstance(response, dict) and "messages" in response:
                logger.info(f"[SearchAgent] Number of messages: {len(response['messages'])}")
                for idx, msg in enumerate(response["messages"]):
                    logger.info(f"[SearchAgent] Message {idx}: type={type(msg)}, has_tool_calls={hasattr(msg, 'tool_calls')}")
                    if hasattr(msg, "type"):
                        logger.info(f"[SearchAgent] Message {idx} type attribute: {msg.type}")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        logger.info(f"[SearchAgent] Message {idx} tool_calls: {msg.tool_calls}")
                    if hasattr(msg, "content"):
                        content_preview = str(msg.content)[:200] if msg.content else "None"
                        logger.info(f"[SearchAgent] Message {idx} content preview: {content_preview}")
            
        except Exception as e:
            logger.error(f"[SearchAgent] Error: {str(e)}")
            return {
                "response": f"Search failed: {str(e)}",
                "error": str(e),
                "urls": [],
                "results": []
            }

        # Extract response text, URLs, and search results
        response_text = self._extract_response_text(response)
        urls, results = self._extract_search_data(response)
        
        # Log the extraction results
        if urls:
            logger.info(f"[SearchAgent] Successfully processed query - URLs: {len(urls)}, Results: {len(results)}")
        else:
            logger.info(f"[SearchAgent] Query processed without web search (answered from context)")
        
        return {
            "response": response_text,
            "urls": urls,
            "results": results
        }

    def _extract_response_text(self, response):
        """Extract text from agent response."""
        if isinstance(response, str):
            return response
        if isinstance(response, dict) and "messages" in response:
            return response["messages"][-1].content
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def _extract_search_data(self, response):
        """Extract URLs and search results from agent response."""
        urls = []
        results = []
        
        try:
            logger.info(f"[SearchAgent] Extracting search data from response")
            
            if isinstance(response, dict) and "messages" in response:
                for idx, message in enumerate(response["messages"]):
                    logger.info(f"[SearchAgent] Processing message {idx}")
                    
                    # Check for tool calls in the message (AI messages)
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        logger.info(f"[SearchAgent] Found {len(message.tool_calls)} tool calls in message {idx}")
                        for tool_call in message.tool_calls:
                            # Store tool call details
                            if hasattr(tool_call, "name"):
                                tool_info = {
                                    "tool": tool_call.name,
                                    "args": tool_call.args if hasattr(tool_call, "args") else {}
                                }
                                logger.info(f"[SearchAgent] Tool call: {tool_info}")
                    
                    # Check for tool messages (responses from tools)
                    if hasattr(message, "type") and message.type == "tool":
                        logger.info(f"[SearchAgent] Found tool message in message {idx}")
                        content = message.content if hasattr(message, "content") else str(message)
                        logger.info(f"[SearchAgent] Tool message content type: {type(content)}")
                        logger.info(f"[SearchAgent] Tool message content preview: {str(content)[:300]}")
                        
                        try:
                            # Try to parse JSON content
                            if isinstance(content, str):
                                parsed_content = json.loads(content)
                                logger.info(f"[SearchAgent] Parsed JSON content type: {type(parsed_content)}")
                                
                                # Handle if parsed_content is a list of search results
                                if isinstance(parsed_content, list):
                                    logger.info(f"[SearchAgent] Found list with {len(parsed_content)} items")
                                    for item in parsed_content:
                                        if isinstance(item, dict):
                                            if "url" in item:
                                                urls.append(item["url"])
                                                logger.info(f"[SearchAgent] Extracted URL: {item['url']}")
                                            results.append(item)
                                
                                # Handle if parsed_content is a dict
                                elif isinstance(parsed_content, dict):
                                    logger.info(f"[SearchAgent] Parsed JSON content keys: {parsed_content.keys()}")
                                    # Extract URLs from search results
                                    if "results" in parsed_content:
                                        logger.info(f"[SearchAgent] Found {len(parsed_content['results'])} results")
                                        for result in parsed_content["results"]:
                                            if "url" in result:
                                                urls.append(result["url"])
                                            results.append(result)
                                    elif "url" in parsed_content:
                                        urls.append(parsed_content["url"])
                                        results.append(parsed_content)
                                    else:
                                        # Store entire parsed content
                                        results.append(parsed_content)
                            
                            elif isinstance(content, dict):
                                # Content is already a dict
                                logger.info(f"[SearchAgent] Content is already dict with keys: {content.keys()}")
                                if "results" in content:
                                    for result in content["results"]:
                                        if "url" in result:
                                            urls.append(result["url"])
                                        results.append(result)
                                elif "url" in content:
                                    urls.append(content["url"])
                                    results.append(content)
                                else:
                                    results.append(content)
                            
                            elif isinstance(content, list):
                                # Content is already a list
                                logger.info(f"[SearchAgent] Content is already list with {len(content)} items")
                                for item in content:
                                    if isinstance(item, dict) and "url" in item:
                                        urls.append(item["url"])
                                    results.append(item)
                        except json.JSONDecodeError as je:
                            logger.warning(f"[SearchAgent] JSON decode error: {str(je)}")
                            # If not JSON, store as text result
                            if content:
                                results.append({"content": str(content)})
            
            logger.info(f"[SearchAgent] Extracted {len(urls)} URLs and {len(results)} results")
        except Exception as e:
            logger.error(f"[SearchAgent] Could not extract search data: {str(e)}", exc_info=True)
        
        return urls, results

# ------------------------- TEST MODE -------------------------

if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from app.config.mongo import init_db, close_db

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise RuntimeError("❌ Missing GROQ_API_KEY in environment variables!")

    # Create LLM (no JSON binding - let agent return natural responses)
    llm = ChatOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0.3,
        model_kwargs={"tool_choice": "auto"}
    )
    
    async def test():
        await init_db()
        
        agent = SearchAgent(llm)
        await agent.initialize()
        
        result = await agent.run("test-user", "test-chat", "What are the latest AI news?")
        print("\n=== SEARCH RESULT ===")
        print(result["response"])
        
        await close_db()

    asyncio.run(test())
