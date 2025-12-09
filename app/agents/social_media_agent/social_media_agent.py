"""
Social Media Agent - LangChain Implementation
Handles Instagram automation using LangChain agent framework
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool

from instagrapi import Client
from instagrapi.exceptions import LoginRequired

from .modules.auth import InstagramAuth
from .modules.posting import InstagramPosting
from .modules.insights import InstagramInsights
from .modules.scraper import InstagramScraper
from .modules.engagement import InstagramEngagement
from .modules.intelligence import ContentIntelligence

logger = logging.getLogger("SocialMediaAgent")


# ===================== INSTAGRAM CLIENT MANAGER =====================

class InstagramClientManager:
    """Manages Instagram client singleton and authentication"""
    
    _instance = None
    _client: Optional[Client] = None
    _username: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._auth = InstagramAuth()
        return cls._instance
    
    def login(self, username: str, password: str, verification_code: Optional[str] = None) -> Client:
        """Login to Instagram and return authenticated client"""
        self._client = self._auth.login(username, password, verification_code)
        self._username = username
        return self._client
    
    def get_client(self) -> Optional[Client]:
        """Get authenticated client"""
        return self._client
    
    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        return self._client is not None
    
    def logout(self):
        """Logout and clear client"""
        if self._client and self._username:
            self._auth.logout()
        self._client = None
        self._username = None


# ===================== TOOL WRAPPERS =====================

client_manager = InstagramClientManager()


def _safe(fn):
    """Ensure tool failures never break the agent."""
    def wrapper(**kwargs):
        try:
            return json.dumps(fn(**kwargs), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    return wrapper


@tool(name_or_callable="instagram_login")
def login_tool(username: str, password: str, verification_code: Optional[str] = None, **_):
    """Login to Instagram account. Required before using other tools."""
    try:
        client_manager.login(username, password, verification_code)
        return json.dumps({"success": True, "message": f"Logged in as {username}"})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool(name_or_callable="instagram_upload_photo")
def upload_photo_tool(photo_path: str, caption: str = "", hashtags: Optional[List[str]] = None, **_):
    """Upload a photo to Instagram feed with caption and hashtags."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    posting = InstagramPosting(client)
    return _safe(posting.upload_photo)(
        photo_path=photo_path,
        caption=caption,
        hashtags=hashtags or []
    )


@tool(name_or_callable="instagram_upload_video")
def upload_video_tool(video_path: str, caption: str = "", hashtags: Optional[List[str]] = None, **_):
    """Upload a video to Instagram feed with caption and hashtags."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    posting = InstagramPosting(client)
    return _safe(posting.upload_video)(
        video_path=video_path,
        caption=caption,
        hashtags=hashtags or []
    )


@tool(name_or_callable="instagram_upload_reel")
def upload_reel_tool(video_path: str, caption: str = "", hashtags: Optional[List[str]] = None, **_):
    """Upload a reel (short video) to Instagram."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    posting = InstagramPosting(client)
    return _safe(posting.upload_reel)(
        video_path=video_path,
        caption=caption,
        hashtags=hashtags or []
    )


@tool(name_or_callable="instagram_get_account_insights")
def get_account_insights_tool(**_):
    """Get account insights including followers, engagement rate, and performance."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    insights = InstagramInsights(client)
    return _safe(insights.get_account_insights)()


@tool(name_or_callable="instagram_get_media_insights")
def get_media_insights_tool(media_id: str, **_):
    """Get detailed insights for a specific post."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    insights = InstagramInsights(client)
    return _safe(insights.get_media_insights)(media_id=media_id)


@tool(name_or_callable="instagram_get_user_profile")
def get_user_profile_tool(username: str, **_):
    """Get detailed profile information for any Instagram user."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    scraper = InstagramScraper(client)
    return _safe(scraper.get_user_profile)(username=username)


@tool(name_or_callable="instagram_get_hashtag_feed")
def get_hashtag_feed_tool(hashtag: str, amount: int = 20, **_):
    """Get recent posts from a hashtag feed."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    scraper = InstagramScraper(client)
    return _safe(scraper.get_hashtag_feed)(hashtag=hashtag, amount=amount)


@tool(name_or_callable="instagram_like_media")
def like_media_tool(media_id: str, **_):
    """Like a post on Instagram."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    engagement = InstagramEngagement(client)
    return _safe(engagement.like_media)(media_id=media_id)


@tool(name_or_callable="instagram_comment_on_media")
def comment_on_media_tool(media_id: str, text: str, **_):
    """Comment on a post on Instagram."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    engagement = InstagramEngagement(client)
    return _safe(engagement.comment_on_media)(media_id=media_id, text=text)


@tool(name_or_callable="instagram_follow_user")
def follow_user_tool(user_id: str, **_):
    """Follow a user on Instagram."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    engagement = InstagramEngagement(client)
    return _safe(engagement.follow_user)(user_id=user_id)


@tool(name_or_callable="instagram_suggest_hashtags")
def suggest_hashtags_tool(caption: Optional[str] = None, niche: Optional[str] = None, count: int = 30, **_):
    """Generate hashtag suggestions based on caption or niche."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    intelligence = ContentIntelligence(client)
    return _safe(intelligence.generate_hashtags)(caption=caption, niche=niche, count=count)


@tool(name_or_callable="instagram_analyze_competitor")
def analyze_competitor_tool(username: str, **_):
    """Analyze a competitor's Instagram account for insights."""
    client = client_manager.get_client()
    if not client:
        return json.dumps({"error": "Not logged in. Use instagram_login first."})
    
    intelligence = ContentIntelligence(client)
    return _safe(intelligence.analyze_competitor)(username=username)


# All available tools
TOOLS = [
    # login_tool,
    # upload_photo_tool,
    # upload_video_tool,
    # upload_reel_tool,
    get_account_insights_tool,
    get_media_insights_tool,
    get_user_profile_tool,
    get_hashtag_feed_tool,
    # like_media_tool,
    # comment_on_media_tool,
    # follow_user_tool,
    suggest_hashtags_tool,
    analyze_competitor_tool,
]


# ===================== SOCIAL MEDIA AGENT =====================

class SocialMediaAgent:
    """
    LangChain-based Social Media Agent for Instagram automation
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = TOOLS
        
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self._system_prompt(),
            middleware=[
                SummarizationMiddleware(
                    model=self.llm,
                    max_tokens_before_summary=4000,
                    messages_to_keep=15
                )
            ]
        )
    
    def _system_prompt(self):
        return """
You are an **Instagram Social Media Manager Agent**.

You help users manage their Instagram account with these capabilities:
- **Analytics**: Get account insights, media insights, performance analysis
- **Research**: Get user profiles, hashtag feeds, competitor analysis
- **Intelligence**: Suggest hashtags, analyze competitors, recommend content

---

### Important Rules
1. **Authentication required**: Always ensure the user is logged in before performing actions
2. **Safe operations**: Respect rate limits and Instagram's terms of service
3. **Clear communication**: Explain what you're doing and provide results clearly
4. **Error handling**: If a tool fails, explain the error and suggest solutions
5. **Privacy**: Never share sensitive credentials in responses

---

### Workflow

1. Check if user is logged in (if not, ask for credentials)
2. Understand user's request
3. Use appropriate tools to fulfill the request
4. Present results in clear, actionable format
5. Suggest next steps if relevant

---

### Output Format

Present results in clear Markdown:

## 📱 Action Taken
Brief description of what was done

## ✅ Results
Key metrics, data, or confirmation

## 💡 Recommendations (optional)
Next steps or insights

---

Be professional, efficient, and helpful!
"""
    
    async def run(self, query: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Execute a user query
        
        Args:
            query: User's request
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Dict with response and metadata
        """
        logger.info(f"[SocialMediaAgent] Processing: {query}")
        
        try:
            result = await self.agent.ainvoke({
                "messages": [
                    {"role": "user", "content": query}
                ],
                "user_id": user_id,
                "session_id": session_id,
            })
            
            # Extract final message
            final_text = result["messages"][-1].content if result.get("messages") else str(result)
            
            return {
                "success": True,
                "response": final_text,
                "metadata": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "authenticated": client_manager.is_authenticated()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error: {str(e)}"
            }
    
    async def stream(self, query: str, user_id: str, session_id: str) -> AsyncGenerator[str, None]:
        """
        Stream agent responses for real-time UI updates
        
        Args:
            query: User's request
            user_id: User identifier
            session_id: Session identifier
            
        Yields:
            Streaming response chunks
        """
        logger.info(f"[SocialMediaAgent STREAM] {query}")
        
        yield "🤖 Processing your request...\n\n"
        
        try:
            async for event in self.agent.astream_events({
                "messages": [{"role": "user", "content": query}],
                "user_id": user_id,
                "session_id": session_id,
            }):
                kind = event.get("event")
                data = event.get("data")
                
                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        yield chunk.content
                
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    yield f"\n🔧 Using tool: {tool_name}\n"
                
                elif kind == "on_tool_end":
                    output = data.get("output", "")
                    if output:
                        yield f"✅ Tool completed\n"
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n❌ Error: {str(e)}\n"


# ===================== STANDALONE USAGE =====================

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test():
        llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.1,
        )
        
        agent = SocialMediaAgent(llm)
        
        # Test query
        result = await agent.run(
            query="Get account insights",
            user_id="test_user",
            session_id="test_session"
        )
        
        print("\n" + "="*50)
        print(result.get("response", ""))
        print("="*50 + "\n")
    
    asyncio.run(test())
