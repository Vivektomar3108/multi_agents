"""
Instagram Agent Configuration
"""
import os
from pathlib import Path


class InstagramConfig:
    """Configuration for Instagram Agent"""
    
    # Session management
    SESSION_DIR = os.getenv("INSTAGRAM_SESSION_DIR", "sessions")
    
    # Rate limiting (actions per hour)
    MAX_LIKES_PER_HOUR = int(os.getenv("MAX_LIKES_PER_HOUR", "60"))
    MAX_COMMENTS_PER_HOUR = int(os.getenv("MAX_COMMENTS_PER_HOUR", "30"))
    MAX_FOLLOWS_PER_HOUR = int(os.getenv("MAX_FOLLOWS_PER_HOUR", "20"))
    MAX_UNFOLLOWS_PER_HOUR = int(os.getenv("MAX_UNFOLLOWS_PER_HOUR", "20"))
    MAX_DMS_PER_HOUR = int(os.getenv("MAX_DMS_PER_HOUR", "10"))
    
    # Safety delays (in seconds)
    DELAY_RANGE_LIKE = (2, 5)
    DELAY_RANGE_COMMENT = (5, 10)
    DELAY_RANGE_FOLLOW = (3, 7)
    DELAY_RANGE_UNFOLLOW = (3, 7)
    DELAY_RANGE_DM = (10, 20)
    
    # Content limits
    MAX_HASHTAGS_PER_POST = 30
    MAX_CAPTION_LENGTH = 2200
    MAX_ALBUM_ITEMS = 10
    
    # Download settings
    DOWNLOAD_DIR = os.getenv("INSTAGRAM_DOWNLOAD_DIR", "downloads")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "instagram_agent.log")
    
    @classmethod
    def get_session_path(cls, username: str) -> Path:
        """Get session file path for username"""
        return Path(cls.SESSION_DIR) / f"{username}_session.json"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        Path(cls.SESSION_DIR).mkdir(exist_ok=True)
        Path(cls.DOWNLOAD_DIR).mkdir(exist_ok=True)


# Safety recommendations
SAFETY_TIPS = [
    "Don't exceed Instagram's daily limits (500-1000 actions per day)",
    "Use random delays between actions",
    "Gradually increase activity on new accounts",
    "Avoid posting the same comment repeatedly",
    "Don't follow/unfollow the same users repeatedly",
    "Engage with content relevant to your niche",
    "Maintain a natural-looking activity pattern",
    "Use session files to avoid frequent logins",
    "Monitor for action blocks and adjust behavior",
    "Comply with Instagram's Terms of Service"
]


# Hashtag categories
HASHTAG_CATEGORIES = {
    "fitness": ["fitness", "gym", "workout", "health", "training"],
    "travel": ["travel", "wanderlust", "adventure", "explore", "vacation"],
    "food": ["food", "foodie", "foodporn", "instafood", "yummy"],
    "fashion": ["fashion", "style", "ootd", "fashionblogger", "outfit"],
    "photography": ["photography", "photo", "photographer", "photooftheday"],
    "business": ["business", "entrepreneur", "marketing", "success", "motivation"]
}
